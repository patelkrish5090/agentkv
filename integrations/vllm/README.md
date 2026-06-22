# vLLM Integration for AgentKV

> **Status: Phase 3 — Not yet implemented.**

---

## Integration Design

AgentKV hooks into vLLM as a custom KV cache allocator.  The exact integration
point depends on the **installed vLLM version**, which must be read from source
before implementation begins (per the project's "no invented APIs" rule).

## How to Find the Integration Point

```bash
# Find the BlockSpaceManager / KV allocator in your vLLM install:
python -c "import vllm; print(vllm.__file__)"
# Then grep for BlockSpaceManager or KVCache in the source tree
```

## Known vLLM Extension Points (as of vLLM 0.4.x)

| Version range | Extension point | Notes |
|---|---|---|
| ≥ 0.4.0 | `BlockSpaceManager` subclass | Register via `--kv-cache-dtype` or monkey-patch |
| ≥ 0.5.0 | Plugin API (`vllm.plugins`) | Cleaner; preferred if available |

> **If the installed version doesn't have a clean plugin API**, the integration
> will use the best available hook and this limitation will be documented here
> rather than forcing a brittle monkey-patch that breaks on vLLM updates.

## Activation

```python
# Planned activation (Phase 3):
from agentkv.integrations.vllm import AgentKVAllocator
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_allocator=AgentKVAllocator(capacity_gb=20, block_size=16),
)
```

## Limitations

- Output correctness parity with the stock allocator is a hard requirement;
  any divergence is a bug.
- See the project README for what AgentKV explicitly does NOT do.
