# Benchmarks

## Headline result: real vLLM engine, real preemption event avoided (2026-07-27)

The most meaningful result so far — not a memory-delta on a bare allocator,
but AgentKV patched into vLLM's *actual* `LLMEngine`/`Scheduler`, compared
against unmodified stock vLLM, on identical concurrent traffic. Reproduce
with `bench/vllm_capacity_stress.py --backend {agentkv,stock} --sweep
4,8,16,32,64` (see `bench/colab_t4_benchmark.ipynb` section 8d for the exact
commands and environment setup — vLLM 0.5.4 requires a specific
torch/triton/xformers stack pinned together, documented there).

**Setup**
- GPU: Tesla T4 (Google Colab), 16 GB VRAM
- Engine: real vLLM v0.5.4 (`LLMEngine` + `Scheduler` + `Worker`), not a stub
- Model: `facebook/opt-125m`
- `gpu_memory_utilization=0.1` — deliberately small so capacity pressure is
  reachable without a huge model or absurd sequence counts. Total pool:
  **1317 GPU KV blocks, fixed and identical for both backends** (vLLM's
  `CacheEngine` reserves this upfront based on `gpu_memory_utilization`,
  independent of which `BlockSpaceManager` governs it)
- Workload: a shared support-AI context (incident history + billing policy)
  + several distinct concurrent user questions, each explored with 4
  parallel completions (`SamplingParams(n=4)`) — swept from 4 to 64
  concurrent users (16 to 256 total sequences)
- `agentkv` backend: real `AgentKVBlockManager` patched in via `patch_vllm()`
- `stock` backend: vLLM's own unmodified `BlockSpaceManagerV2`
- Metric: peak GPU KV blocks actually in use, polled directly from the
  scheduler's live `block_manager.get_num_free_gpu_blocks()` during
  `generate()` — a real read of vLLM's own state, not inferred

**Measured peak KV blocks used (of 1317 total)**

| n_users | total sequences | AgentKV | Stock | Blocks saved | % saved |
|---|---|---|---|---|---|
| 4  | 16  | 100 (7.6%)   | 128 (9.7%)    | 28  | 21.9% |
| 8  | 32  | 176 (13.4%)  | 260 (19.7%)   | 84  | 32.3% |
| 16 | 64  | 327 (24.8%)  | 507 (38.5%)   | 180 | 35.5% |
| 32 | 128 | 617 (46.8%)  | 1037 (78.7%)  | 420 | **40.5%** |
| 64 | 256 | 1189 (90.3%) | 1314 (99.8%)  | 125 | 9.5% |

**The result that matters most isn't even in that table.** During the stock
run at `n_users=32`, vLLM's own scheduler logged this, unprompted:

```
WARNING: Sequence group 123 is preempted by PreemptionMode.SWAP mode because
there is not enough KV cache space. This can affect the end-to-end performance.
```

Stock hit real memory pressure and preempted a live request — a genuine
operational cost (vLLM's own warning says so) — at a concurrency level where
AgentKV, serving the identical workload, was sitting at 46.8% utilization
with no preemption anywhere in its log at any point in the sweep. Stock
suffered a real, measured consequence at a load AgentKV had more than half
its capacity left to absorb.

**Honest read of the curve:** savings peak in the middle (40.5% at
`n_users=32`) and compress at the top (`n_users=64`, both backends near the
fixed pool's ceiling — 90.3% vs 99.8%). That's expected, not a flaw: once
both allocators are nearly full, there's little room left for any allocator
to differentiate. The more informative reading is that AgentKV likely
absorbs meaningfully more concurrent load before hitting the wall stock was
already hitting at `n_users=64`, not that the advantage "disappears" at scale.

**Not yet measured:** the actual maximum concurrent users each backend can
sustain before failing outright (this sweep stopped at 64, chosen in
advance, not because either backend broke); throughput (tokens/sec) under
this same concurrent load; nested/multi-round forking (Tree-of-Thought
depth > 1); stock vLLM with its own native `--enable-prefix-caching` turned
on instead of left at its default off (see the follow-up result below for
why this matters).

## Confirmed on the real target-scale model: DeepSeek-R1-Distill-Llama-8B (2026-07-27)

The result above used `facebook/opt-125m` to keep the first real-engine
attempt small and debuggable. This repeats the identical capacity-sweep
methodology on the actual 8B model used elsewhere in this project (section 6
of the notebook) — confirming the mechanism generalizes past a toy model.

**Setup:** same as above, except model = `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
loaded via vLLM's own in-flight bitsandbytes 4-bit quantization (~5.34 GB
weights), `gpu_memory_utilization=0.7`, `max_model_len=4096` (capped down
from the checkpoint's inherited 128K default — this benchmark's prompts
never approach that length, and vLLM refuses to start otherwise since it
insists the reserved KV cache be able to hold at least one full-length
sequence). Total pool: **1239 GPU KV blocks, fixed for both backends.**

**Measured peak KV blocks used (of 1239 total)**

| n_users | total sequences | AgentKV | Stock | Blocks saved | % saved |
|---|---|---|---|---|---|
| 2  | 8  | 56 (4.5%)   | 56 (4.5%)   | 0   | 0% |
| 4  | 16 | 103 (8.3%)  | 116 (9.4%)  | 13  | 11.2% |
| 8  | 32 | 163 (13.2%) | 228 (18.4%) | 65  | 28.5% |
| 16 | 64 | 287 (23.2%) | 456 (36.8%) | 169 | **37.1%** |

Same shape as the opt-125m result: negligible difference at the lowest
concurrency (`n_users=2` — both backends handle a single request's own
`n=4` parallel-sampling fork equally well, since that's built into vLLM's
own stock block manager too, not something exclusive to AgentKV), growing
to 37.1% fewer blocks at `n_users=16`. The real differentiator is
cross-request sharing between *different* users' prompts (which share the
common context but diverge at the question) — that's the part stock isn't
doing here.

**Important caveat, not hidden:** both runs show `enable_prefix_caching=False`
in vLLM's own logged config — stock vLLM ships with its own optional native
prefix-caching feature *off* by default, so this comparison is "AgentKV vs.
vLLM's out-of-the-box configuration," not "AgentKV vs. vLLM's own caching
turned on." AgentKV's cross-request sharing is unconditional (it doesn't
read or respect that flag), so a stricter follow-up would re-run stock with
`--enable-prefix-caching` on to see whether AgentKV still wins against
vLLM's own native answer to the same problem. That comparison hasn't been
run yet.

## AgentKVCache vs. naive copy-per-agent, real model, real GPU (2026-07-26)

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

## Older synthetic-shape results

`bench/microbench_alloc.py` and `bench/branching_benchmark.py --mode internal`
have also been run for real on the same T4 (real Triton kernels, real CUDA
allocation), but against synthetic KV shapes with no actual model attached —
useful for allocator-latency and CoW-bookkeeping numbers, not representative
of a real serving workload. See `bench/results/` for raw output from a given
run.
