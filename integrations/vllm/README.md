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
