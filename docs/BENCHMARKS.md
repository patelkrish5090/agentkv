# Benchmarks

## Real result: AgentKVCache real-GPU CoW fork (2026-07-26)

First real-hardware, real-model result — measured from `torch.cuda.memory_allocated()`,
not computed from a formula. Reproduce with `bench/hf_cache_demo.py` (see
`bench/colab_t4_benchmark.ipynb` section 6 for the exact Colab setup).

**Setup**
- GPU: Tesla T4 (Google Colab), 16 GB VRAM
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, 4-bit NF4 (bitsandbytes)
- AgentKV pool: 1024 blocks × 16 tokens, 32 layers × 8 KV heads × 128 head_dim, float16 (2.00 GiB)
- Workload: 1 shared prompt (66 tokens) → forked into 4 agents → 60 new tokens generated each, greedy decoding
- Cache backend: `agentkv.hf_cache.AgentKVCache` as `past_key_values` — the real
  GPU storage `model.generate()` reads/writes, not a side tracker alongside a
  plain HF cache

**Measured VRAM (`torch.cuda.memory_allocated()`)**

| Stage | VRAM |
|---|---|
| After model load | 5.71 GB |
| After AgentKV pool init | 7.85 GB (+2.15 GB — the pool itself) |
| After prefill (66-token shared prompt) | 7.86 GB |
| **After forking 4 agents** | **7.86 GB (+0.000 GB)** |
| After generating 60 tokens × 4 agents | 7.86 GB |
| Peak during generation | 8.13 GB |

**Takeaway:** forking 4 agents from the same prefix cost measured **zero
additional VRAM** — `AgentKVPool.fork()` is a ref-count bump, confirmed on
real hardware, not just in the CPU-only test suite. The *absolute* savings
here are small in GB terms because the shared prompt is short (66 tokens) —
there just isn't much KV to share yet. Savings scale with shared-prefix
length × agent count; a longer shared system prompt / RAG context would show
a larger absolute number. What this result establishes is correctness of the
mechanism on real hardware, not a throughput/memory-ceiling claim.

**Not yet measured:** an equivalent naive (copy-per-agent) baseline on the
same real model, to turn the above into a percentage-savings number; and any
result at longer shared-prefix lengths or higher branching factors.

## Older synthetic-shape results

`bench/microbench_alloc.py` and `bench/branching_benchmark.py --mode internal`
have also been run for real on the same T4 (real Triton kernels, real CUDA
allocation), but against synthetic KV shapes with no actual model attached —
useful for allocator-latency and CoW-bookkeeping numbers, not representative
of a real serving workload. See `bench/results/` for raw output from a given
run; no aggregate numbers are published here yet pending a second pass with a
longer shared prefix.
