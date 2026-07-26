"""Stub of sglang.srt.mem_cache.cache_init_params.CacheInitParams — trimmed
to the fields integrations/sglang/radix_cache.py reads."""

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass
class CacheInitParams:
    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    page_size: int = 1
    disable: bool = False
    max_context_len: Optional[int] = None
