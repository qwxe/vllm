# vLLM fork dev recipe (v0.19.0, no CUDA compile)

Repo: /home/asd/vllm-dev (shallow, tag v0.19.0 = 2a69949). Branch `holes/base` = pristine v0.19.0.
Worktrees (one per agent): /home/asd/vllm-dev-wt/{test,core,kernel,cp} on branches holes/{test,core,kernel,cp}.
Venv (torch 2.10+cu128, vllm 0.19.0 wheel, all deps): /home/asd/LLMServingSim/.wt/vllm-venv

## How it works
Each worktree holds the pure-Python source of v0.19.0. The compiled files the git tree lacks
(`vllm/*.abi3.so`, `vllm/vllm_flash_attn/*`, `_version.py`, ...) are copied from the wheel with
`rsync --ignore-existing` (already done for all four worktrees). Selecting a worktree is just
`PYTHONPATH=<worktree>`; the venv python then imports that tree's `vllm` instead of the wheel's.
Python edits take effect immediately. C++/CUDA (csrc/) changes are NOT picked up (precompiled .so);
kernel work must be in Python/Triton, or needs a real build.

## Allowed command forms (use exactly these prefixes)
New worktree:
    git -C /home/asd/vllm-dev worktree add -b holes/<b> /home/asd/vllm-dev-wt/<b> holes/base
    rsync -a --ignore-existing --exclude __pycache__ /home/asd/LLMServingSim/.wt/vllm-venv/lib/python3.12/site-packages/vllm/ /home/asd/vllm-dev-wt/<b>/vllm/
Git inside a worktree:
    git -C /home/asd/vllm-dev-wt/<b> status | diff | add | commit ...
Run on the GPU (always via gpurun, serialises GPU use):
    /home/asd/LLMServingSim/.wt/bin/gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/<b> /home/asd/LLMServingSim/.wt/vllm-venv/bin/python /path/to/script.py > /path/to/log 2>&1
Pure CPU import check:
    /home/asd/LLMServingSim/.wt/vllm-venv/bin/python -c "..."   (with PYTHONPATH via `env` if allowed)
The overlaid .so files are untracked in git (shown by `git status`); do not commit them.

## Gotchas
- Scripts using `LLM(...)` MUST have `if __name__ == "__main__":` (engine core is spawned; otherwise RuntimeError).
- Confirm the right tree is used: `print(vllm.__file__)` should be /home/asd/vllm-dev-wt/<b>/vllm/__init__.py.
- Pipe output to a log file and grep it; vLLM logs are long.

## GPU memory
RTX 4050 Laptop, 6 GiB, ~0.8 GiB used by the display, ~4.95 GiB free. vLLM requires free >= util * total,
so gpu_memory_utilization=0.9 FAILS at startup. Use <= 0.75 (0.7 verified). Use enforce_eager=True, max_model_len=2048.
Host RAM is 12 GB: one heavy process at a time (gpurun waits for MemAvailable >= 3000 MB).

## Test model
Qwen/Qwen2.5-0.5B-Instruct downloads fine from the HF Hub and gives correct text
("The capital of France is" -> " Paris. It is the largest city in"). Llama is gated (needs HF_TOKEN).
Backend chosen by vLLM on this GPU: FLASH_ATTN.

## Verified smoke
Reference script: /home/asd/vllm-dev-wt/test_fwd.py (env TEST_MODEL, LOAD_FORMAT=dummy optional).
    /home/asd/LLMServingSim/.wt/bin/gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/test /home/asd/LLMServingSim/.wt/vllm-venv/bin/python /home/asd/vllm-dev-wt/test_fwd.py > /home/asd/vllm-dev-wt/test_fwd.log 2>&1
Result: imported from the test worktree, vllm 0.19.0, one generation OK (~2 min incl. startup).
