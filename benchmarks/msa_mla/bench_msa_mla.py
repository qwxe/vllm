# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA_MLA vs stock TRITON_MLA prefill on holed steps (one MLA layer).

Drives both backends exactly as a holed vLLM step does: a request-level
CommonAttentionMetadata is split into segments by holes/core
(`build_hole_segments` + `segment_common_attn_metadata`), each backend's own
metadata builder builds its metadata, and `forward_mha` (the prefill) runs on a
paged latent cache. DeepSeek-V2-Lite dims (H 16, latent 512 + rope 64, qk 192,
v 128), bf16, block 16, random weights.

Per layout: correctness of both against a dense fp32 reference (E <= 32K),
GPU time of forward_mha (L2 flushed before every call, median over rounds,
round-robin), host time of the metadata build.

    python benchmarks/msa_mla/bench_msa_mla.py MODEL_DIR out.jsonl [--quick]
MODEL_DIR: any DeepSeek-V2(-Lite) config dir (only the config is read).
"""

import json
import math
import os
import sys
import tempfile
import time

import numpy as np
import torch

H, P, R, V, L = 16, 128, 64, 128, 512
BS = 16
_CTX = None


def setup(model_dir, max_len):
    from vllm.config import (CacheConfig, CompilationConfig, DeviceConfig, LoadConfig,
                             ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig)
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)

    mc = ModelConfig(model=model_dir, tokenizer=model_dir, dtype="bfloat16", seed=0,
                     max_model_len=max_len)
    cc = CacheConfig(block_size=BS, cache_dtype="auto")
    cc.num_gpu_blocks = 1000
    cc.num_cpu_blocks = 0
    sc = SchedulerConfig(max_num_seqs=32, max_num_batched_tokens=16384,
                         enable_chunked_prefill=True, max_model_len=max_len,
                         is_encoder_decoder=False)
    vc = VllmConfig(model_config=mc, cache_config=cc, parallel_config=ParallelConfig(),
                    scheduler_config=sc, device_config=DeviceConfig(),
                    load_config=LoadConfig(), compilation_config=CompilationConfig())
    global _CTX
    ctx = _CTX = set_current_vllm_config(vc)  # keep alive: GC would close it
    ctx.__enter__()
    f = tempfile.mkstemp()[1]
    init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"file://{f}",
                                 local_rank=0, backend="nccl")
    initialize_model_parallel(1, 1)
    return vc


def make_impl(backend, vc, kv_b_proj, scale):
    impl_cls = backend.get_class().get_impl_cls()
    impl = impl_cls(num_heads=H, head_size=L + R, scale=scale, num_kv_heads=1,
                    alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
                    logits_soft_cap=None, attn_type="decoder",
                    kv_sharing_target_layer_name=None, q_lora_rank=None, kv_lora_rank=L,
                    qk_nope_head_dim=P, qk_rope_head_dim=R, qk_head_dim=P + R,
                    v_head_dim=V, kv_b_proj=kv_b_proj)
    impl.dcp_world_size = 1
    return impl


def layouts(quick):
    out = []
    Es = [16384, 32768] if quick else [16384, 32768, 65536]
    for E in Es:
        t = 2048
        fr = E - t
        out.append((f"nohole tail2048 @{E // 1024}K", E, fr, [], t))
        out.append((f"rate: hole 2048 @{E // 8 // 1024}K + tail2048 @{E // 1024}K", E, fr,
                    [(E // 8, E // 8 + 2048)], t))
        for h in (1, 4, 16):
            lo, hi = E // 2, fr - 64
            st = [lo + (hi - lo) * i // h for i in range(h)]
            out.append((f"{h} x q16 deep + tail2048 @{E // 1024}K", E, fr,
                        [(s, s + 16) for s in st], t))
        for h in (4, 16):
            lo, hi = E // 8, fr - 512
            st = [lo + (hi - lo) * i // h for i in range(h)]
            out.append((f"{h} x q256 + tail2048 @{E // 1024}K", E, fr,
                        [(s, s + 256) for s in st], t))
        for h in (1, 2, 4, 8, 16):
            lo, hi = E // 2, E - 64
            st = [lo + (hi - lo) * i // h for i in range(h)]
            out.append((f"hole-only {h} x q16 deep @{E // 1024}K", E, E,
                        [(s, s + 16) for s in st], 0))
        out.append((f"hole-only q2048 @{E // 2048}K of {E // 1024}K", E, E,
                    [(E // 2, E // 2 + 2048)], 0))
    out.append(("fresh 8192", 8192, 0, [], 8192))
    return out


def main():
    model_dir, out_path = sys.argv[1], sys.argv[2]
    quick = "--quick" in sys.argv
    rounds = 3 if quick else 7
    dev = torch.device("cuda")
    vc = setup(model_dir, 65536 + 4096)
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as BE
    from vllm.v1.attention.ops.msa.api_mla import mla_reference
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.hole_segments import (build_hole_segments,
                                              segment_common_attn_metadata)

    torch.manual_seed(0)
    dt = torch.bfloat16
    w = (torch.randn(L, H * (P + V), device=dev) / math.sqrt(L)).to(dt)
    kv_b_proj = ColumnParallelLinear(input_size=L, output_size=H * (P + V), bias=False).to(dev, dt)
    kv_b_proj.weight = torch.nn.Parameter(w.T.contiguous(), requires_grad=False)
    scale = 1.0 / math.sqrt(P + R)
    spec = MLAAttentionSpec(block_size=BS, num_kv_heads=1, head_size=L + R, dtype=dt)
    names = ["model.layers.0.self_attn.attn"]
    backends = {"TRITON_MLA": BE.TRITON_MLA, "MSA_MLA": BE.MSA_MLA}
    impls = {k: make_impl(b, vc, kv_b_proj, scale) for k, b in backends.items()}
    builders = {k: b.get_class().get_builder_cls()(spec, names, vc, dev)
                for k, b in backends.items()}
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    k_scale = torch.tensor(1.0, device=dev)
    fout = open(out_path, "w")

    for name, E, frontier, holes, n_tail in layouts(quick):
        m = sum(e - s for s, e in holes) + n_tail
        nblk = -(-max(E, frontier + n_tail) // BS)
        cache = torch.randn(nblk + 8, BS, L + R, device=dev).to(dt)
        bt = torch.randperm(nblk + 8, device=dev)[:nblk].to(torch.int32)[None, :]
        positions = (frontier + np.arange(m)).astype(np.int64)
        seg = build_hole_segments(["r"], {"r": holes} if holes else {}, np.array([m]),
                                  np.array([frontier]), positions)
        pos_t = torch.from_numpy(positions).to(dev)
        slots = (bt[0, pos_t // BS].long() * BS + pos_t % BS)
        seq_len = frontier + n_tail
        cm = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, m], dtype=torch.int32, device=dev),
            query_start_loc_cpu=torch.tensor([0, m], dtype=torch.int32),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=dev),
            num_reqs=1, num_actual_tokens=m, max_query_len=m, max_seq_len=seq_len,
            block_table_tensor=bt, slot_mapping=slots,
            _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
            _num_computed_tokens_cpu=torch.tensor([frontier], dtype=torch.int32))
        if holes:
            cm = segment_common_attn_metadata(cm, seg, dev)
        q = torch.randn(m, H, P + R, device=dev).to(dt)
        new = cache.view(-1, L + R)[slots]
        kv_c, k_pe = new[:, :L].contiguous(), new[:, None, L:].contiguous()
        rec = dict(layout=name, E=E, m=m, n_seg=int(seg.num_segments))
        outs, times, host = {}, {k: [] for k in backends}, {}
        variants = list(backends) + ["MSA_MLA:absorb", "MSA_MLA:compute"]
        if "--alias" in sys.argv and holes:
            # the profiler's view: segments as separate requests with aliased
            # block tables, no hole_segments
            variants += ["MSA_MLA:alias", "MSA_MLA:alias-nogroup"]
            cm_alias = cm.replace(hole_segments=None)
        mds = {}
        for k in variants:
            bk = "MSA_MLA" if k.startswith("MSA_MLA") else k
            b = builders[bk]
            if k == "MSA_MLA:absorb":
                b.path = "absorb"
            elif k == "MSA_MLA:compute":
                b.path = "compute"
            elif k.startswith("MSA_MLA:alias"):
                b.path = "auto"
                b.group_aliased = k == "MSA_MLA:alias"
            if bk == "MSA_MLA" and not k.startswith("MSA_MLA:alias"):
                b.group_aliased = False
            elif bk == "MSA_MLA":
                b.path = "auto"
            t0 = time.perf_counter()
            mds[k] = b.build(0, cm_alias if k.startswith("MSA_MLA:alias") else cm)
            torch.cuda.synchronize()
            host[k] = (time.perf_counter() - t0) * 1e3
            times.setdefault(k, [])
        # skip the absorbed variant where it is hopeless (big m) to keep runs short
        if m > 4096:
            variants.remove("MSA_MLA:absorb")
        if m > 1024:
            variants.remove("MSA_MLA:compute")
        for r in range(rounds):
            for k in variants:
                impl = impls["MSA_MLA" if k.startswith("MSA_MLA") else k]
                o = torch.empty(m, H * V, device=dev, dtype=dt)
                flush.zero_()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                impl.forward_mha(q, kv_c, k_pe, cache, mds[k], k_scale, o)
                e.record()
                torch.cuda.synchronize()
                times[k].append(s.elapsed_time(e) * 1e3)
                if r == 0:
                    outs[k] = o.view(m, H, V).float()
        if E <= 32768:
            # dense reference over [0, E) of this request
            kp = torch.arange(E, device=dev)
            lat = cache.view(-1, L + R)[bt[0, kp // BS].long() * BS + kp % BS].float()
            kvn = (lat[:, :L] @ w.float()).view(E, H, P + V)
            o_ref, _ = mla_reference(q[..., :P], q[..., P:], kvn[..., :P], lat[:, L:],
                                     kvn[..., P:], pos_t, torch.zeros_like(pos_t), kp,
                                     torch.zeros_like(kp), scale)
            rec["err"] = {k: float((o - o_ref).abs().max()) for k, o in outs.items()}
        rec["us"] = {k: float(np.median(v)) for k, v in times.items() if v}
        rec["host_ms"] = host
        rec["msa_stats"] = mds["MSA_MLA"].prefill.stats
        st = rec["us"]
        print(f"{name:45s} m={m:5d} seg={rec['n_seg']:2d} stock={st['TRITON_MLA']:10.0f} "
              f"msa={st['MSA_MLA']:10.0f} absorb={st.get('MSA_MLA:absorb', float('nan')):10.0f} "
              f"compute={st.get('MSA_MLA:compute', float('nan')):10.0f} "
              f"x={st['TRITON_MLA'] / st['MSA_MLA']:.2f} err={rec.get('err')}"
              + (f" alias={st.get('MSA_MLA:alias'):.0f} alias_nogroup="
                 f"{st.get('MSA_MLA:alias-nogroup'):.0f}" if 'MSA_MLA:alias' in st else ""),
              flush=True)
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        del cache, mds, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
