"""Stub of sglang.srt.mem_cache.memory_pool.ReqToTokenPool — trimmed to the
surface integrations/sglang/radix_cache.py touches."""

from typing import Any, List


class ReqToTokenPool:
    def __init__(self, size: int, max_context_len: int) -> None:
        self.size = size
        self.max_context_len = max_context_len

    def available_size(self) -> int:
        return self.size

    def write(self, indices, values) -> None:
        pass

    def alloc(self, reqs: List[Any]):
        return list(range(len(reqs)))

    def free(self, req: Any) -> None:
        pass

    def clear(self) -> None:
        pass
