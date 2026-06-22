# SGLang Integration for AgentKV

> **Status: Phase 3 — Not yet implemented.**

---

## Integration Design

AgentKV hooks into SGLang's `RadixAttention` cache pool.  The actual extension
point is confirmed by reading SGLang's installed source before writing any code.

## How to Find the Integration Point

```bash
python -c "import sglang; print(sglang.__file__)"
# Grep for RadixAttention, cache_pool, or alloc in the source
```

## Known SGLang Extension Points (as of SGLang 0.2.x)

| Component | Class/function | Notes |
|---|---|---|
| RadixAttention | `RadixCache` in `sglang.srt.mem_pool` | Primary hook target |
| Allocator | `ReqToTokenPool` / `TokenToKVPool` | Block-level replacement |

## Activation (Planned)

```python
from agentkv.integrations.sglang import AgentKVCacheHook
import sglang as sgl

sgl.set_default_backend(sgl.Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    cache_hook=AgentKVCacheHook(capacity_gb=20, block_size=16),
))
```

## Limitations

- Same correctness guarantee as the vLLM integration: identical output tokens
  with AgentKV enabled vs. disabled is a hard requirement.
- If SGLang's RadixAttention is not pluggable in the installed version, the
  best available hook will be used and the limitation documented here.
