"""Stub of sglang.srt.mem_cache.base_prefix_cache — trimmed to the fields
and methods integrations/sglang/radix_cache.py actually implements against.

Field names match the real dataclasses (SGLang main, commit 10908a679,
2026-07-18). Fields AgentKV's adapter doesn't use (mamba/SWA/hierarchical
host-cache variants) are kept only where needed for signature compatibility,
defaulted to None/0/empty so callers that don't care about them don't need
to pass them.
"""

import dataclasses
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence

import torch


@dataclasses.dataclass
class RadixKey:
    token_ids: List[int]
    extra_key: Optional[str] = None

    def __len__(self) -> int:
        return len(self.token_ids)


@dataclasses.dataclass
class MatchPrefixParams:
    key: RadixKey
    cow_mamba: bool = False
    req: Optional[Any] = None


class MatchResult:
    def __init__(
        self,
        device_indices: torch.Tensor,
        last_device_node: Any = None,
        last_host_node: Any = None,
        best_match_node: Any = None,
        host_hit_length: int = 0,
        full_kv_hit_length: int = 0,
    ) -> None:
        self.device_indices = device_indices
        self.last_device_node = last_device_node
        self.last_host_node = last_host_node
        self.best_match_node = best_match_node
        self.host_hit_length = host_hit_length
        self.full_kv_hit_length = full_kv_hit_length


@dataclasses.dataclass
class InsertParams:
    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None
    prev_prefix_len: int = 0
    chunked: bool = False
    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None


@dataclasses.dataclass
class EvictParams:
    num_tokens: int = 0


@dataclasses.dataclass
class EvictResult:
    num_tokens_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    delta: Optional[int] = None


@dataclasses.dataclass
class DecLockRefParams:
    pass


@dataclasses.dataclass
class DecLockRefResult:
    delta: Optional[int] = None


class BasePrefixCache(ABC):
    req_to_token_pool: Any
    token_to_kv_pool_allocator: Any
    page_size: int
    disable: bool

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult: ...

    @abstractmethod
    def cache_finished_req(self, req: Any, is_insert: bool = True, **kwargs): ...

    @abstractmethod
    def cache_unfinished_req(self, req: Any, **kwargs): ...

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult: ...

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult: ...

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult: ...

    def evictable_size(self) -> int:
        return 0

    def protected_size(self) -> int:
        return 0

    def total_size(self) -> int:
        return self.evictable_size() + self.protected_size()

    def is_chunk_cache(self) -> bool:
        return False
