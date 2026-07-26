# vLLM Integration for AgentKV

AgentKV provides a drop-in integration for vLLM that replaces the default `BlockSpaceManager` with `AgentKVBlockManager`. 

By patching vLLM to use AgentKV, you benefit from:
1. **O(1) Branching**: Zero-copy CoW branching using `DualRadixTree`.
2. **Lock-Free Sharing**: Built-in deterministic prefix sharing based on actual Agent sequences instead of background content-hash matching.

## Usage

You must call `patch_vllm()` *before* initializing the `LLM` or `AsyncLLMEngine`.

```python
import vllm
from integrations.vllm import patch_vllm

# Patch vLLM's BlockSpaceManagerV2 with AgentKV
patch_vllm()

# Initialize vLLM normally. 
# It will internally instantiate AgentKVBlockManager for the KV Cache.
llm = vllm.LLM(model="facebook/opt-125m")

# Generation works natively
outputs = llm.generate(["A long time ago in a galaxy far, far away..."])
print(outputs[0].outputs[0].text)
```

## How it Works

Under the hood, `AgentKVBlockManager` subclasses `vllm.core.interfaces.BlockSpaceManager`.
1. **Zero VRAM Allocation**: AgentKV's physical pool (`SlabAllocator`) is initialized on the `meta` device. This ensures it consumes 0 bytes of real VRAM. vLLM's `CacheEngine` continues to allocate and manage the actual physical GPU KV tensors.
2. **CoW Hooks**: `AgentKVPool` intercepts any required CoW operations during `append_tokens` and passes them up to vLLM's `append_slots` loop, which executes the physical `CacheEngine.copy()` operations efficiently.

## Known limitations (v1)

- **Prefix sharing is bounded by concurrently-alive sequences, not time.**
  `commit_prefix()` promotes a prompt's blocks into AgentKV's shared tree so
  *other currently-running* sequences can match it, but the shared node is
  pruned the instant the last sequence referencing it calls `free()` (see
  `DualRadixTree._dec_shared_ref_locked`). This is not a time-persistent
  cross-request cache the way vLLM's own automatic prefix caching is — it
  covers the branching/concurrent-agent case AgentKV targets (many
  sequences alive at once sharing a prefix), not reuse against requests
  that fully finished earlier with nothing else still referencing them.
- **No swap support.** `swap_in`/`swap_out` raise `NotImplementedError` —
  matches AgentKV's v1 scope (no CPU offload).
- **Not yet run against a real vLLM install.** vLLM requires Triton + a CUDA
  build not available in this dev environment; this adapter is validated
  against a local stub of vLLM's `core.interfaces`/`sequence` modules
  (`tests/fakes/vllm/`, signatures copied from the vendored `vllm-src/`
  reference tree) — see `tests/test_vllm_integration.py`.
