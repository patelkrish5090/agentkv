"""
agentkv integrations — SGLang prefix-cache backend.

Registers AgentKVRadixCache (DualRadixTree-backed) as an SGLang
mem_cache backend via sglang.srt.mem_cache.registry. See README.md in
this directory for usage and known limitations.

``AgentKVRadixCache`` and ``register`` are re-exported lazily via
``__getattr__`` so importing this package doesn't require SGLang to be
installed unless you actually use them (mirrors integrations/vllm).
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("AgentKVRadixCache", "register"):
        from . import radix_cache

        return getattr(radix_cache, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentKVRadixCache", "register"]
