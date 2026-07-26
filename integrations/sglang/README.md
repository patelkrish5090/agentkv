# SGLang Integration for AgentKV

AgentKV provides an `AgentKVRadixCache` backend for SGLang, registered through
SGLang's mem_cache backend registry (`sglang.srt.mem_cache.registry`) — no
monkey-patching required.

## Usage

```python
import sglang
from integrations.sglang import register

# Register "agentkv" as an available prefix-cache backend before launching.
register()
```

Then launch the SGLang server with `--radix-cache-backend agentkv` (or set
the equivalent field on `server_args` when embedding SGLang directly).

## What this replaces, and what it doesn't

SGLang's own RadixAttention already performs the physical KV cache writes
through `token_to_kv_pool_allocator` / `req_to_token_pool` — this adapter
never touches those. What `AgentKVRadixCache` owns instead is the *tree
metadata* SGLang's scheduler needs:

- `match_prefix` — longest shared-prefix lookup for a new request
- `cache_unfinished_req` / `cache_finished_req` — keep the tree in sync as
  requests generate tokens, and promote finished requests' tokens into the
  shared tree
- `evict` / `inc_lock_ref` / `dec_lock_ref` — eviction bookkeeping

There's no explicit `fork()` in SGLang's interface the way vLLM has one:
sharing between requests happens purely through repeated `match_prefix`
calls against the same tree — AgentKV's own `fork` mechanism, minus the
handle.

## Why this exists now (history)

An earlier version of this README argued that, since SGLang's RadixAttention
already does prefix caching and CoW natively, hooking AgentKV into it would
be "functionally redundant" and left this integration unimplemented. That
argument undersold what AgentKV adds: SGLang's registry is a first-class
extension point specifically for swapping in a different tree/allocator
implementation, and AgentKV's DualRadixTree is a genuine (if today unproven)
alternative worth being able to benchmark against RadixAttention head-to-head
— which requires it to actually exist as running code, not just an argument
for why it wouldn't be worth building.

## Known limitations (v1)

- **Prefix sharing is bounded by concurrently-alive requests, not
  time.** AgentKV's shared-tree node is pruned the instant its last
  referencing handle frees (see `DualRadixTree._dec_shared_ref_locked`) —
  there is no idle, time-persistent prefix cache the way SGLang's own
  RadixAttention eviction policy provides. This adapter is well-suited to
  AgentKV's target case — many concurrently active branching agents sharing
  a prefix *while they're all still running* — but will not, by itself,
  reuse a prefix from a request that has already fully finished and freed
  with nothing else still referencing it.
- **Page-level granularity only.** A matched prefix is rounded down to whole
  pages; up to one page's worth of an otherwise-matching prefix may be
  reported as a miss. Same tradeoff the vLLM integration makes.
- **No hierarchical/host-memory cache, SWA, or mamba support** — same v1
  scope limit as the rest of AgentKV (single-node, GPU-resident only).
- **Not yet run against real SGLang.** SGLang requires a CUDA build not
  available in this dev environment; this adapter is validated against a
  local stub of SGLang's mem_cache interface (`tests/fakes/sglang/`,
  signatures pulled from SGLang `main` @ commit `10908a679`, 2026-07-18) —
  see `tests/test_sglang_integration.py`. SGLang's internal mem_cache API
  moves fast; re-verify signatures in
  `sglang.srt.mem_cache.base_prefix_cache` / `registry` before relying on
  this against a specific pinned SGLang release.

## Difference between AgentKV and SGLang RadixAttention

RadixAttention is SGLang's own built-in prefix-cache/CoW system, tightly
coupled to SGLang's serving stack. AgentKV is a framework-agnostic memory
pool that can be dropped into HuggingFace `transformers.Cache`, vLLM's
`BlockSpaceManager`, or (via this adapter) SGLang's mem_cache registry —
useful if you want one allocator's behavior and metrics across frameworks
rather than each framework's native implementation.
