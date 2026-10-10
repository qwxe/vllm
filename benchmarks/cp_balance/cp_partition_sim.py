"""Context-parallel (CP) partition of one request's prefill work over d ranks.

A prefill step of a request computes a set of position ranges: hole pieces
(KV missing inside a reused prefix) and the tail past the frontier. Under
causal attention a token at position i costs about ``c(i) = a + b * i``, so
equal token counts are not equal work once the ranges sit at different
depths. Three partitioners:

- ``contig``: the computed tokens laid out in position order, cut into d
  contiguous pieces of equal token count.
- ``zigzag``: the standard causal split over the same concatenated tokens:
  2d equal-token chunks, rank r gets chunks r and 2d-1-r. It assumes one
  contiguous sequence starting at 0, so holes and a deep start skew it.
- ``zigzag_abs``: the zigzag over absolute positions: [0, end) is cut into
  2d equal-length position chunks, and rank r computes the tokens of its two
  chunks that are in a computed range.
- ``balanced`` (the CP-balance kernel): the cumulative cost F over the
  computed tokens in position order, cut into d equal-cost pieces by binary
  search on the prefix sums. A cut may split a range.

All functions are pure. Ranges are ``(start, length)`` pairs; the result per
rank is a list of ``(start, length)`` pieces in position order. The cost of a
range [s, e) is the integral of c over it, ``a (e - s) + b (e^2 - s^2) / 2``,
the same shape as the prefill work fit.
"""

PARTITIONERS = ("contig", "zigzag", "zigzag_abs", "balanced")

# Defaults: the prefill work fit (seconds per token, seconds per token^2).
DEFAULT_A = 3.87e-5
DEFAULT_B = 1.85e-9


def range_cost(s, e, a=DEFAULT_A, b=DEFAULT_B):
    """Cost of computing positions [s, e)."""
    if e <= s:
        return 0.0
    return a * (e - s) + b * (e * e - s * s) / 2.0


def _normalize(ranges):
    """Drop empty ranges and sort by start. Ranges must not overlap."""
    out = sorted((int(s), int(n)) for s, n in ranges if n > 0)
    for (s0, n0), (s1, _) in zip(out, out[1:]):
        if s0 + n0 > s1:
            raise ValueError(f"[cp_partition] overlapping ranges {out}")
    return out


def _slice(ranges, lo, hi):
    """Pieces covering concatenated-token indices [lo, hi) of ``ranges``."""
    pieces = []
    off = 0
    for s, n in ranges:
        a, b = max(lo, off), min(hi, off + n)
        if a < b:
            pieces.append((s + a - off, b - a))
        off += n
        if off >= hi:
            break
    return pieces


def _equal_token_cuts(total, parts):
    """Token indices cutting ``total`` tokens into ``parts`` near-equal pieces."""
    return [total * j // parts for j in range(parts + 1)]


def _contig(ranges, d, a, b):
    total = sum(n for _, n in ranges)
    cuts = _equal_token_cuts(total, d)
    return [_slice(ranges, cuts[r], cuts[r + 1]) for r in range(d)]


def _zigzag(ranges, d, a, b):
    total = sum(n for _, n in ranges)
    cuts = _equal_token_cuts(total, 2 * d)
    out = []
    for r in range(d):
        j = 2 * d - 1 - r
        out.append(_slice(ranges, cuts[r], cuts[r + 1]) + _slice(ranges, cuts[j], cuts[j + 1]))
    return out


def _zigzag_abs(ranges, d, a, b):
    if not ranges:
        return [[] for _ in range(d)]
    end = ranges[-1][0] + ranges[-1][1]
    cuts = _equal_token_cuts(end, 2 * d)

    def clip(lo, hi):
        pieces = []
        for s, n in ranges:
            x, y = max(lo, s), min(hi, s + n)
            if x < y:
                pieces.append((x, y - x))
        return pieces

    out = []
    for r in range(d):
        j = 2 * d - 1 - r
        out.append(clip(cuts[r], cuts[r + 1]) + clip(cuts[j], cuts[j + 1]))
    return out


def _balanced(ranges, d, a, b):
    total = sum(n for _, n in ranges)
    if total == 0:
        return [[] for _ in range(d)]
    # Prefix cost at each range boundary, in concatenated-token order.
    offs, pre = [0], [0.0]
    for s, n in ranges:
        offs.append(offs[-1] + n)
        pre.append(pre[-1] + range_cost(s, s + n, a, b))

    def F(t):
        """Cost of the first t computed tokens."""
        # The range holding token t (linear: a request has few ranges).
        i = 0
        while i + 1 < len(ranges) and offs[i + 1] < t:
            i += 1
        s = ranges[i][0]
        return pre[i] + range_cost(s, s + (t - offs[i]), a, b)

    C = pre[-1]
    cuts = [0]
    for j in range(1, d):
        target = C * j / d
        lo, hi = cuts[-1], total
        # Smallest t with F(t) >= target, then the nearer of t-1 and t.
        while lo < hi:
            mid = (lo + hi) // 2
            if F(mid) < target:
                lo = mid + 1
            else:
                hi = mid
        t = lo
        if t > cuts[-1] and target - F(t - 1) < F(t) - target:
            t -= 1
        cuts.append(t)
    cuts.append(total)
    return [_slice(ranges, cuts[r], cuts[r + 1]) for r in range(d)]


_IMPL = {"contig": _contig, "zigzag": _zigzag, "zigzag_abs": _zigzag_abs, "balanced": _balanced}


def partition(ranges, d, method="balanced", a=DEFAULT_A, b=DEFAULT_B):
    """Split ``ranges`` ((start, length) pairs) over ``d`` ranks.

    Returns ``(rank_pieces, rank_costs)``: per rank, its ``(start, length)``
    pieces in position order and their summed cost. Every computed token goes
    to exactly one rank.
    """
    if method not in _IMPL:
        raise ValueError(f"[cp_partition] unknown partitioner {method!r}; one of {PARTITIONERS}")
    d = int(d)
    if d < 1:
        raise ValueError(f"[cp_partition] cp size must be >= 1, got {d}")
    ranges = _normalize(ranges)
    if d == 1:
        pieces = [list(ranges)]
    else:
        pieces = [sorted(p) for p in _IMPL[method](ranges, d, a, b)]
    costs = [sum(range_cost(s, s + n, a, b) for s, n in p) for p in pieces]
    return pieces, costs


def imbalance(costs):
    """max rank cost / mean rank cost; 1.0 for no work."""
    total = sum(costs)
    if total <= 0:
        return 1.0
    return max(costs) * len(costs) / total


def _selftest():
    """Runs with a plain ``python3 -m serving.core.cp_partition``."""
    def check(ranges, d, method):
        pieces, costs = partition(ranges, d, method)
        flat = sorted(p for ps in pieces for p in ps)
        # Coverage: every computed position exactly once.
        want = sorted(i for s, n in ranges for i in range(s, s + n))
        got = sorted(i for s, n in flat for i in range(s, s + n))
        assert got == want, (method, ranges, d, pieces)
        assert len(pieces) == d
        return pieces, costs

    # One contiguous prefill from 0: zigzag and balanced both near 1, contig worse.
    one = [(0, 65536)]
    for d in (2, 4, 8):
        _, cc = check(one, d, "contig")
        _, cz = check(one, d, "zigzag")
        _, cb = check(one, d, "balanced")
        assert imbalance(cb) < 1.0001, cb
        assert imbalance(cz) < 1.01, cz
        assert imbalance(cc) > imbalance(cz), (cc, cz)
    # zigzag == zigzag_abs when the computed tokens are exactly [0, end).
    assert partition(one, 4, "zigzag")[0] == partition(one, 4, "zigzag_abs")[0]

    # Holes deep in a reused prefix plus a tail: zigzag over the concatenation
    # is skewed, balanced is not.
    holes = [(20000, 512), (61440, 1024), (90000, 256), (100000, 3000)]
    for d in (2, 4):
        _, cz = check(holes, d, "zigzag")
        _, cza = check(holes, d, "zigzag_abs")
        _, cb = check(holes, d, "balanced")
        assert imbalance(cb) <= min(imbalance(cz), imbalance(cza)) + 1e-9, (cb, cz, cza)
        assert imbalance(cb) < 1.001, cb

    # Degenerate inputs.
    p, c = check([], 4, "balanced")
    assert p == [[], [], [], []] and imbalance(c) == 1.0
    p, c = check([(5, 3)], 4, "balanced")       # fewer tokens than ranks
    assert sum(len(x) for x in p) <= 3
    p, c = check([(7, 9)], 1, "zigzag")
    assert p == [[(7, 9)]]
    for m in PARTITIONERS:
        check([(0, 1), (10, 1), (100, 1)], 2, m)
        check([(16, 4096), (8192, 2048)], 3, m)
    print("cp_partition selftest: ok")


if __name__ == "__main__":
    _selftest()
