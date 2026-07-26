"""Stub of sglang.srt.mem_cache.allocator.base.BaseTokenToKVPoolAllocator —
trimmed to what integrations/sglang/radix_cache.py reads from it (page_size,
available_size / total size bookkeeping). The real class also owns alloc()/
free() for physical KV slots, which AgentKV's adapter deliberately does not
touch — see integrations/sglang/README.md."""

import abc


class BaseTokenToKVPoolAllocator(abc.ABC):
    def __init__(self, size: int, page_size: int) -> None:
        self.size = size
        self.page_size = page_size

    def available_size(self) -> int:
        return self.size

    @abc.abstractmethod
    def clear(self): ...

    @abc.abstractmethod
    def alloc(self, need_size: int): ...

    @abc.abstractmethod
    def free(self, free_index): ...


class FakePagedAllocator(BaseTokenToKVPoolAllocator):
    """Concrete stand-in with a trivial bump allocator, enough to exercise
    AgentKVRadixCache's bookkeeping in tests without a real GPU KV pool."""

    def clear(self):
        pass

    def alloc(self, need_size: int):
        return None

    def free(self, free_index):
        pass
