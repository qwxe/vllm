# SPDX-License-Identifier: Apache-2.0
"""gloo test of the CP_HOLES backend glue (metadata builder + impl).

Drives `CPHolesMetadataBuilder.build` and `CPHolesImpl.forward` in the
replicated-dense mode (the stock model runner feeds every CP rank all tokens)
on a paged KV cache laid out exactly as vLLM's PCP slot mapping lays it out,
and checks every rank's full output against single-device attention, plus the
cache writes. Needs vLLM installed (imports vllm.v1.attention.backend).

    python tests/v1/attention/cp_holes/test_backend_gloo.py --world 2 4
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)


def _import_backend():
    """The backend module, from the installed vLLM or the staging tree."""
    try:
        from vllm.v1.attention.backends import cp_holes as be
        return be
    except ImportError:
        pass
    sys.path.insert(0, os.path.join(ROOT, "vllm", "v1", "attention"))
    import cp_holes  # noqa: F401  (staging copy)
    for name, mod in list(sys.modules.items()):
        if name == "cp_holes" or name.startswith("cp_holes."):
            sys.modules["vllm.v1.attention." + name] = mod
    path = os.path.join(ROOT, "vllm", "v1", "attention", "backends",
                        "cp_holes.py")
    spec = importlib.util.spec_from_file_location(
        "vllm.v1.attention.backends.cp_holes", path)
    be = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = be
    spec.loader.exec_module(be)
    return be


def _worker(rank, world, port, scheme_name, q):
    try:
        torch.set_num_threads(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["VLLM_CP_HOLES_SCHEME"] = scheme_name
        os.environ["VLLM_CP_HOLES_DENSE"] = "replicated"
        os.environ["VLLM_CP_HOLES_PASSQ_RATIO"] = "0.35"
        dist.init_process_group("gloo", rank=rank, world_size=world)
        be = _import_backend()
        import test_cp_gloo as T
        cph = T.cph
        hq, hkv, dh, u = 8, 4, 16, 16
        plan0, efull = T.build_case(4242 + world, world, u, 6)
        segs = plan0.segments
        n = plan0.num_reqs
        e_step = plan0.e_step
        Q, K, V = T.make_tensors(plan0, efull, hq, hkv, dh, 1, torch.float32,
                                 99)
        scale = 1.0 / math.sqrt(dh)
        ref_o, _ = T.reference(plan0, efull, Q, K, V, scale)[0]

        sharded = scheme_name in ("passq", "ring", "adaptive")
        w, r = (world, rank) if sharded else (1, 0)
        if scheme_name == "ulysses":
            _, ks = cph.schemes.ulysses_heads(world, hq, hkv, rank)
        else:
            ks = slice(0, hkv)
        hkv_st = ks.stop - ks.start
        # global block table: same logical block ids on every rank
        nblk = [-(-e // (u * w)) for e in efull]
        mx = max(nblk)
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(sum(nblk) + 2, generator=g)
        bt = torch.zeros(n, mx, dtype=torch.int32)
        off = 0
        for i, nb in enumerate(nblk):
            bt[i, :nb] = perm[off:off + nb].to(torch.int32)
            off += nb
        cache = torch.zeros(2, len(perm), u, hkv_st, dh)
        flat = cache.flatten(1, 2)
        for i in range(n):  # resident keys at vLLM's slots
            comp = plan0.computed_mask(i)
            pos = torch.arange(efull[i])
            res = torch.ones(efull[i], dtype=torch.bool)
            res[:comp.numel()] = ~comp
            sm = cph.kv_store.vllm_slot_mapping(bt[i], pos, u, r, w, u)
            sel = res & (sm >= 0)
            flat[0, sm[sel]] = K[0][i][pos[sel]][:, ks]
            flat[1, sm[sel]] = V[0][i][pos[sel]][:, ks]

        # stock per-token inputs: request-major, segments in position order
        qlens = [sum(ln for _, ln in s) for s in segs]
        qsl = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)))
        cam = SimpleNamespace(
            num_reqs=n, num_actual_tokens=plan0.m,
            seq_lens_cpu=torch.tensor(e_step), query_start_loc_cpu=qsl,
            block_table_tensor=bt, cp_segments=segs)
        mc = SimpleNamespace(get_num_attention_heads=lambda pc: hq,
                             get_num_kv_heads=lambda pc: hkv,
                             get_head_size=lambda: dh)
        cfg = SimpleNamespace(model_config=mc, parallel_config=SimpleNamespace(
            cp_kv_cache_interleave_size=u))
        builder = be.CPHolesMetadataBuilder(SimpleNamespace(block_size=u),
                                            ["l0"], cfg, torch.device("cpu"))
        builder.comm = be.CPComm(dist.group.WORLD, rank, world)
        md = builder.build(0, cam)
        impl = be.CPHolesImpl(hq, dh, scale, hkv)
        k_all = torch.stack([K[0][int(plan0.qreq[x])][int(plan0.qpos[x])]
                             for x in range(plan0.m)])
        v_all = torch.stack([V[0][int(plan0.qreq[x])][int(plan0.qpos[x])]
                             for x in range(plan0.m)])
        out = torch.empty(plan0.m, hq * dh)
        impl.forward(None, Q[0].reshape(plan0.m, -1), k_all.reshape(
            plan0.m, -1), v_all.reshape(plan0.m, -1), cache, md, output=out)
        err = (out.view(plan0.m, hq, dh) - ref_o).abs().max().item()
        # cache writes: computed positions this rank holds are now in place
        for i in range(n):
            pos = torch.arange(e_step[i])[plan0.computed_mask(i)]
            sm = cph.kv_store.vllm_slot_mapping(bt[i], pos, u, r, w, u)
            sel = sm >= 0
            assert torch.equal(flat[0, sm[sel]], K[0][i][pos[sel]][:, ks])
            assert torch.equal(flat[1, sm[sel]], V[0][i][pos[sel]][:, ks])
        q.put((rank, err, None))
        dist.barrier()
        dist.destroy_process_group()
    except Exception:
        q.put((rank, None, traceback.format_exc()))


def run(world, scheme_name):
    import test_cp_gloo as T
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = T._free_port()
    ps = [ctx.Process(target=_worker, args=(r, world, port, scheme_name, q))
          for r in range(world)]
    for p in ps:
        p.start()
    res = [q.get(timeout=900) for _ in range(world)]
    for p in ps:
        p.join(timeout=60)
    for _, _, tb in res:
        if tb:
            raise AssertionError(tb)
    return max(e for _, e, _ in res)


def test_backend_glue():
    for w in (2, 4):
        for s in ("ulysses", "passq", "ring", "allgather", "adaptive"):
            assert run(w, s) < 2e-5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, nargs="+", default=[2])
    ap.add_argument("--schemes", nargs="+", default=[
        "ulysses", "passq", "ring", "allgather", "adaptive"])
    a = ap.parse_args()
    for w in a.world:
        for s in a.schemes:
            e = run(w, s)
            print(f"{'OK ' if e < 2e-5 else 'BAD'} backend world={w} {s:9s} "
                  f"max_err={e:.2e}", flush=True)
