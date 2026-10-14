"""Build a small DeepSeek-V2-Lite slice with real weights: first L layers,
first NE routed experts, vocab truncated to NV ids. Fetches only the needed
bytes by HTTP range requests and streams them to disk (no large RAM use)."""
import json, os, struct, sys, urllib.request
REPO = "deepseek-ai/DeepSeek-V2-Lite"
OUT, L, NE, NV = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
os.makedirs(OUT, exist_ok=True)
base = f"https://huggingface.co/{REPO}/resolve/main/"
ESZ = {"BF16": 2, "F16": 2, "F32": 4}

def get(name, rng=None):
    req = urllib.request.Request(base + name)
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]-1}")
    return urllib.request.urlopen(req, timeout=600)

idx = json.loads(get("model.safetensors.index.json").read())["weight_map"]
def keep(k):
    if k.startswith("model.layers."):
        parts = k.split(".")
        if int(parts[2]) >= L:
            return False
        if ".mlp.experts." in k and int(parts[5]) >= NE:
            return False
    return True
want = sorted(k for k in idx if keep(k))
hdrs = {}
for shard in sorted({idx[k] for k in want}):
    n = struct.unpack("<Q", get(shard, (0, 8)).read())[0]
    hdrs[shard] = (n, json.loads(get(shard, (8, 8 + n)).read()))
plan, off = {}, 0
for k in want:
    n, hdr = hdrs[idx[k]]
    h = hdr[k]
    shape = list(h["shape"])
    rows = shape[0]
    if k in ("model.embed_tokens.weight", "lm_head.weight"):
        rows = min(rows, NV)
    elif k.endswith("mlp.gate.weight"):
        rows = min(rows, NE)
    row_bytes = ESZ[h["dtype"]]
    for d in shape[1:]:
        row_bytes *= d
    nbytes = rows * row_bytes
    a = h["data_offsets"][0]
    plan[k] = (idx[k], 8 + n + a, nbytes, h["dtype"], [rows] + shape[1:])
    off += nbytes
meta, o = {}, 0
for k in want:
    meta[k] = {"dtype": plan[k][3], "shape": plan[k][4], "data_offsets": [o, o + plan[k][2]]}
    o += plan[k][2]
hb = json.dumps(meta).encode()
hb += b" " * ((8 - len(hb) % 8) % 8)
print("total", off / 2**20, "MiB", len(want), "tensors", flush=True)
with open(os.path.join(OUT, "model.safetensors.part"), "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for i, k in enumerate(want):
        shard, start, nbytes, _, _ = plan[k]
        r = get(shard, (start, start + nbytes))
        got = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            got += len(b)
        assert got == nbytes, (k, got, nbytes)
        if i % 40 == 0:
            print(i, len(want), k, flush=True)
os.rename(os.path.join(OUT, "model.safetensors.part"), os.path.join(OUT, "model.safetensors"))
cfg = json.loads(get("config.json").read())
cfg.update(num_hidden_layers=L, n_routed_experts=NE, vocab_size=NV)
cfg.pop("auto_map", None)
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print("done", flush=True)
