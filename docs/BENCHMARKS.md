# Benchmarks

## Real result: AgentKV vs. naive copy-per-agent, real model, real GPU (2026-07-26)

Real-hardware, real-model, **with a real naive baseline** — every number below
is `torch.cuda.memory_allocated()`, measured from two separate process runs
(fresh CUDA context each — the `agentkv` and `naive` backends were never run
in the same process, so there's no risk of one run's allocator state leaking
into the other's numbers). Reproduce with `bench/hf_cache_demo.py --backend
{agentkv,naive} --prompt-preset long --n-agents 8` (see
`bench/colab_t4_benchmark.ipynb` section 6b/6c/6d for the exact commands).

**Setup**
- GPU: Tesla T4 (Google Colab), 16 GB VRAM
- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, 4-bit NF4 (bitsandbytes)
- Shared prompt: 879 tokens (`--prompt-preset long`, a genuine multi-section
  clinical-guideline context, not padding/repetition)
- Workload: 1 shared prefix -> forked/cloned into 8 agents -> 60 new tokens
  generated each, greedy decoding
- `agentkv` backend: `agentkv.hf_cache.AgentKVCache` as `past_key_values` —
  the real GPU storage `model.generate()` reads/writes; forking is a real
  `AgentKVPool.fork()` ref-count bump
- `naive` backend: plain HF `DynamicCache`, and forking is a real
  `copy.deepcopy()` of the whole cache per agent — the actual thing every
  naive per-request KV allocation scheme is doing, not a formula standing in
  for it

**Measured VRAM (`torch.cuda.memory_allocated()`, GB)**

| Stage | naive | agentkv |
|---|---|---|
| After model load | 5.707 | 5.707 |
| After cache/pool init | 5.707 | 7.855 *(fixed 2 GiB pool reservation — see note)* |
| After prefill | 6.060 | 7.863 |
| **After forking 8 agents** | **6.982** | **7.863** |
| After generating 60 tok × 8 agents | 7.075 | 7.864 |
| Peak during generation | 7.131 | 8.133 |

**Headline numbers (deltas — the honest comparison, not the raw totals)**

| Metric | naive | agentkv | Result |
|---|---|---|---|
| Cost of forking 8 agents (prefill -> after-fork delta) | **+0.922 GB** | **+0.000 GB** | **100% of fork-time memory cost eliminated** |
| Total growth, prefill -> peak-during-generation | +1.071 GB | +0.270 GB | **74.8% reduction** |

**Why deltas, not raw totals:** AgentKV's raw numbers (7.8–8.1 GB) look
bigger than naive's (6–7 GB) at a glance. That's not AgentKV using more
memory for the workload — it's a fixed 2 GiB pool pre-allocated upfront
(only 107/1024 blocks, ~10%, actually used by this workload). This is a
deliberate design choice (no dynamic `cudaMalloc` in the hot path — see main
README), and the same reservation would serve many more agents without
growing further, unlike naive which keeps growing linearly per agent. The
delta numbers above are what's actually comparable; reading the raw totals
alone would be misleading in AgentKV's favor... or against it, depending on
which stage you compare, which is exactly why they're reported separately
here instead of as one blended "savings %" that hides the pool-reservation
effect.

**Also real, at the shorter/smaller end:** the same `agentkv` backend was
run earlier at a much shorter shared prompt (66 tokens, 4 agents) and also
showed a measured **+0.000 GB** fork cost — the mechanism holds regardless
of prompt length; what changes with prompt length is how large an absolute
number the *avoided* cost is.

**Not yet measured:** throughput (tokens/sec) comparison between backends;
behavior at higher agent counts or multiple rounds of forking (nested
branching, e.g. Tree-of-Thought depth > 1); any result on a model other than
DeepSeek-R1-Distill-Llama-8B.

## Older synthetic-shape results

`bench/microbench_alloc.py` and `bench/branching_benchmark.py --mode internal`
have also been run for real on the same T4 (real Triton kernels, real CUDA
allocation), but against synthetic KV shapes with no actual model attached —
useful for allocator-latency and CoW-bookkeeping numbers, not representative
of a real serving workload. See `bench/results/` for raw output from a given
run.
