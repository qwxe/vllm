# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA_MLA_CP on a real-weight DeepSeek-V2 slice, d ranks sharing one GPU
(gloo), without a vLLM engine.

Every rank is one process that does what a PCP worker of vLLM does in the
replicated dense mode: it builds the vLLM model (vLLM's DeepseekV2 modules,
real weights from the slice, vLLM's model loader), binds a paged MLA cache,
and for each step builds the step's CommonAttentionMetadata (request-level,
then holes/core's `segment_common_attn_metadata` when the step has holes),
the PCP slot mapping (positions other ranks hold get -1) and the attention
metadata through the backend's own builder, then runs the model forward under
vLLM's forward context and samples greedily. The PCP group is vLLM's own
(`initialize_model_parallel(prefill_context_model_parallel_size=d)`), on gloo;
the device communicator (PyNCCL, which refuses two ranks on one GPU) is
disabled, and the backend's collectives stage through the host.

Scenario (two requests, decodes first in every batch, as vLLM orders them):
prefill A in two chunks next to B's prefill, decode, then evict hole ranges
(cache slots overwritten with garbage on every rank) and run one holed step
(A: two holes + a 3-token hole, compute path, B: a front hole, absorbed path;
both with a 1-token tail), then decode again. Ranks compare their greedy
tokens and logits with a reference saved by `--ref` (d = 1 with the no-CP
MSA_MLA backend, the same scenario).

    # reference (no CP, MSA_MLA), then CP at d = 2 and 4
    python test_mla_cp_model_gloo.py --model <slice> --world 1 --backend MSA_MLA \
        --out ref.pt
    python test_mla_cp_model_gloo.py --model <slice> --world 2 --ref ref.pt
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import traceback

import torch
import torch.multiprocessing as mp

PROMPT_A, PROMPT_B = 1500, 700


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def prompt_ids(model: str, n: int, seed: int) -> list[int]:
    import random

    vocab = 16384
    try:
        from tokenizers import Tokenizer

        import this  # noqa: F401  (Zen of Python, natural text)
        zen = "".join({**{chr(i + 65): chr((i + 13) % 26 + 65) for i in range(26)},
                       **{chr(i + 97): chr((i + 13) % 26 + 97) for i in range(26)}}.get(c, c)
                      for c in this.s)
        tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
        text = "".join(f"Section {k}. " + zen.splitlines()[2 + k % 19] + f" (note {k * 7 % 13})\n"
                       for k in range(300))
        ids = [t for t in tok.encode(text).ids if t < vocab]
        off = (seed * 977) % (len(ids) - n - 1)
        return ids[off:off + n]
    except Exception:  # noqa: BLE001  (no tokenizer: random ids)
        rng = random.Random(seed)
        return [rng.randrange(100, vocab) for _ in range(n)]


class MiniRunner:
    """The parts of the GPU model runner a CP step needs, for one rank."""

    def __init__(self, args, rank: int, world: int):
        from vllm.config import set_current_vllm_config
        from vllm.distributed import parallel_state as ps
        from vllm.engine.arg_utils import EngineArgs

        dist_backend = getattr(args, "dist_backend", "gloo")
        if dist_backend == "gloo":
            # PyNCCL refuses two ranks on one GPU: keep the groups on gloo only
            orig = ps.init_model_parallel_group

            def no_dev_comm(*a, **kw):
                kw["use_device_communicator"] = False
                return orig(*a, **kw)

            ps.init_model_parallel_group = no_dev_comm
        u = args.u
        extra = {}
        if getattr(args, "load_format", None):
            extra["load_format"] = args.load_format
        if getattr(args, "hf_overrides", None):
            import json

            extra["hf_overrides"] = json.loads(args.hf_overrides)
        max_len = getattr(args, "max_model_len", 4096)
        ea = EngineArgs(
            model=args.model, dtype=args.dtype, block_size=u, enforce_eager=True,
            max_model_len=max_len, max_num_batched_tokens=max(4096, max_len),
            max_num_seqs=8,
            prefill_context_parallel_size=world, cp_kv_cache_interleave_size=u,
            cp_kv_placement="interleave", attention_backend=args.backend,
            distributed_executor_backend="external_launcher" if world > 1 else None,
            compilation_config={"custom_ops": ["-rotary_embedding"]},
            enable_prefix_caching=False, seed=0, **extra)
        self.cfg = cfg = ea.create_engine_config()
        # the whole worker runs inside the config context, as vLLM's workers do
        self._cfg_ctx = set_current_vllm_config(cfg)
        self._cfg_ctx.__enter__()
        # ranks share the GPUs round-robin (all on one GPU here)
        local = rank % torch.cuda.device_count()
        self.dev = torch.device(f"cuda:{local}")
        torch.cuda.set_device(self.dev)
        ps.init_distributed_environment(
            world_size=world, rank=rank, local_rank=local, backend=dist_backend,
            distributed_init_method=f"tcp://127.0.0.1:{args.port}")
        ps.initialize_model_parallel(prefill_context_model_parallel_size=world,
                                     backend=dist_backend)
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.model_loader import get_model

        self.model = get_model(vllm_config=cfg)
        layers = {k: v for k, v in cfg.compilation_config.static_forward_context.items()
                  if isinstance(v, MLAAttention)}
        self.layer_names = list(layers)
        spec = layers[self.layer_names[0]].get_kv_cache_spec(cfg)
        backend = layers[self.layer_names[0]].attn_backend
        self.builder = backend.get_builder_cls()(spec, self.layer_names, cfg, self.dev)
        self.rank, self.world, self.u, self.bs = rank, world, u, spec.block_size
        self.num_blocks = getattr(args, "num_blocks", 512)
        dt = cfg.model_config.dtype
        for name, layer in layers.items():
            layer.kv_cache = torch.zeros(self.num_blocks, self.bs, spec.head_size,
                                         dtype=dt, device=self.dev)
        self.layers = layers
        # logical blocks (block_size * world tokens) per request, the same
        # block ids on every rank (the scheduler's)
        self.tables: dict[str, torch.Tensor] = {}
        self.next_block = 1

    def alloc(self, req: str, max_len: int):
        nb = -(-max_len // (self.bs * self.world))
        g = torch.Generator().manual_seed(len(self.tables) + 7)
        ids = self.next_block + torch.randperm(nb, generator=g)
        self.next_block += nb
        assert self.next_block <= self.num_blocks
        self.tables[req] = ids.to(torch.int32)

    def garbage(self, req: str, lo: int, hi: int):
        """Overwrite the cached latent of positions [lo, hi) (an eviction)."""
        from vllm.v1.attention.cp_holes.kv_store import vllm_slot_mapping

        pos = torch.arange(lo, hi)
        sm = vllm_slot_mapping(self.tables[req], pos, self.bs, self.rank, self.world, self.u)
        sm = sm[sm >= 0].to(self.dev)
        for layer in self.layers.values():
            flat = layer.kv_cache.view(-1, layer.kv_cache.shape[-1])
            flat[sm] = 37.0

    @torch.inference_mode()
    def step(self, entries, seqs):
        """entries: [(req, [(start, n), ...])] in batch order (decodes first);
        the segments of a request are its hole pieces in position order, then
        the tail. seqs[req] = all tokens known so far. Returns logits of each
        request's last token."""
        import numpy as np

        from vllm.forward_context import set_forward_context
        from vllm.v1.attention.backend import CommonAttentionMetadata
        from vllm.v1.attention.cp_holes.kv_store import vllm_slot_mapping
        from vllm.v1.worker.hole_segments import HoleSegments, segment_common_attn_metadata

        dev = self.dev
        pos, ids, qlens, ends, sms = [], [], [], [], []
        sreq, sq, se, hole_tok = [], [], [], []
        for i, (req, segs) in enumerate(entries):
            p = torch.cat([torch.arange(s, s + n) for s, n in segs])
            pos.append(p)
            ids.append(torch.tensor([seqs[req][x] for x in p.tolist()]))
            qlens.append(int(p.numel()))
            ends.append(segs[-1][0] + segs[-1][1])
            sms.append(vllm_slot_mapping(self.tables[req], p, self.bs, self.rank,
                                         self.world, self.u))
            for s, n in segs:
                sreq.append(i)
                sq.append(n)
                se.append(s + n)
            hole_tok.append(sum(n for _, n in segs[:-1]))
        n_req = len(entries)
        maxb = max(int(t.numel()) for t in self.tables.values())
        bt = torch.zeros(n_req, maxb, dtype=torch.int32)
        for i, (req, _) in enumerate(entries):
            bt[i, :self.tables[req].numel()] = self.tables[req]
        qsl = torch.zeros(n_req + 1, dtype=torch.int32)
        qsl[1:] = torch.tensor(qlens).cumsum(0)
        seq_lens = torch.tensor(ends, dtype=torch.int32)
        positions = torch.cat(pos).to(dev)
        slot_mapping = torch.cat(sms).to(dev)
        cm = CommonAttentionMetadata(
            query_start_loc=qsl.to(dev), query_start_loc_cpu=qsl,
            seq_lens=seq_lens.to(dev), num_reqs=n_req,
            num_actual_tokens=int(qsl[-1]), max_query_len=max(qlens),
            max_seq_len=max(ends), block_table_tensor=bt.to(dev),
            slot_mapping=slot_mapping, _seq_lens_cpu=seq_lens,
            _num_computed_tokens_cpu=seq_lens - torch.tensor(qlens, dtype=torch.int32))
        if any(hole_tok):
            hs = HoleSegments(seg_req_idx=np.asarray(sreq, dtype=np.int32),
                              seg_query_lens=np.asarray(sq, dtype=np.int32),
                              seg_ends=np.asarray(se, dtype=np.int32),
                              req_hole_tokens=np.asarray(hole_tok, dtype=np.int32))
            cm = segment_common_attn_metadata(cm, hs, dev)
        md = self.builder.build(0, cm)
        attn_md = {n: md for n in self.layer_names}
        sm = {n: slot_mapping for n in self.layer_names}
        with set_forward_context(attn_md, self.cfg, num_tokens=int(qsl[-1]), slot_mapping=sm):
            hidden = self.model(input_ids=torch.cat(ids).to(dev), positions=positions)
        last = (qsl[1:] - 1).to(dev, torch.int64)
        return self.model.compute_logits(hidden[last]).float()


def scenario(run: MiniRunner, model: str, decode_steps: int = 4):
    A, B = "A", "B"
    seqs = {A: prompt_ids(model, PROMPT_A, 1), B: prompt_ids(model, PROMPT_B, 2)}
    run.alloc(A, 4096)
    run.alloc(B, 4096)
    toks, logits = [], []

    def sample(entries, out):
        for (req, segs), lg in zip(entries, out):
            if segs[-1][0] + segs[-1][1] == len(seqs[req]):
                t = int(lg.argmax())
                seqs[req].append(t)
                toks.append((req, t))
                logits.append(lg.cpu())

    def decode():
        e = [(B, [(len(seqs[B]) - 1, 1)]), (A, [(len(seqs[A]) - 1, 1)])]
        sample(e, run.step(e, seqs))

    e = [(A, [(0, 1024)]), (B, [(0, PROMPT_B)])]
    sample(e, run.step(e, seqs))
    e = [(B, [(PROMPT_B, 1)]), (A, [(1024, PROMPT_A - 1024)])]
    sample(e, run.step(e, seqs))
    for _ in range(decode_steps):
        decode()
    # evictions: A two holes + a tiny one (compute path), B a front hole
    # (absorbed path: 64 + 1 tokens <= MSA_MLA_ABSORB_MAX)
    ha = [(256, 512), (800, 832), (1100, 1103)]
    hb = [(0, 64)]
    for lo, hi in ha:
        run.garbage(A, lo, hi)
    for lo, hi in hb:
        run.garbage(B, lo, hi)
    e = [(B, [(lo, hi - lo) for lo, hi in hb] + [(len(seqs[B]) - 1, 1)]),
         (A, [(lo, hi - lo) for lo, hi in ha] + [(len(seqs[A]) - 1, 1)])]
    sample(e, run.step(e, seqs))
    for _ in range(decode_steps):
        decode()
    return toks, torch.stack(logits)


def worker(rank, world, args, q):
    try:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
        os.environ["WORLD_SIZE"] = str(world)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(args.port)
        if args.workspace:
            os.environ["MSA_MLA_WORKSPACE"] = str(args.workspace)
        run = MiniRunner(args, rank, world)
        toks, logits = scenario(run, args.model)
        q.put((rank, toks, logits, None))
    except Exception:  # noqa: BLE001
        q.put((rank, None, None, traceback.format_exc()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/asd/vllm-dev-wt/mla_assets/dsv2lite-1L")
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--backend", default="MSA_MLA_CP")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--u", type=int, default=16, help="block size = interleave size")
    ap.add_argument("--workspace", type=int, default=0,
                    help="MSA_MLA_WORKSPACE (context tokens per pass; 0 = vLLM's)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    args.port = free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, args.world, args, q)) for r in range(args.world)]
    for p in ps:
        p.start()
    print("pids", [p.pid for p in ps], flush=True)
    res = sorted([q.get() for _ in range(args.world)], key=lambda x: x[0])
    for p in ps:
        p.join()
    ok = True
    for rank, _, _, tb in res:
        if tb:
            print(f"rank {rank} failed:\n{tb}", flush=True)
            ok = False
    if not ok:
        print("RESULT FAIL")
        return 1
    toks0, lg0 = res[0][1], res[0][2]
    for rank, toks, lg, _ in res[1:]:
        same = toks == toks0 and torch.equal(lg, lg0)
        print(f"rank {rank} identical to rank 0: {same}", flush=True)
        ok &= same
    print(f"world={args.world} backend={args.backend} u={args.u} dtype={args.dtype} "
          f"workspace={args.workspace or 'default'} tokens={toks0}", flush=True)
    if args.out:
        torch.save({"toks": toks0, "logits": lg0}, args.out)
    if args.ref:
        ref = torch.load(args.ref)
        tok_eq = ref["toks"] == toks0
        agree = next((i for i, (a, b) in enumerate(zip(ref["toks"], toks0)) if a != b),
                     len(toks0))
        d = (lg0 - ref["logits"]).abs().max(dim=-1).values
        lp0 = torch.log_softmax(lg0, -1)
        lpr = torch.log_softmax(ref["logits"], -1)
        dlp = max(float((lp0[i, t] - lpr[i, t]).abs()) for i, (_, t) in enumerate(ref["toks"]))
        print(f"vs ref: tokens_equal={tok_eq} agree={agree}/{len(toks0)} "
              f"max|dlogit| per step={[round(float(x), 6) for x in d]} "
              f"max|dlogprob(ref token)|={dlp:.3e}", flush=True)
        ok &= tok_eq and dlp < args.tol
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
