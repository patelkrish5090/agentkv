"""Stub of sglang.srt.mem_cache.registry — the real plugin point a custom
prefix-cache backend registers against (SGLang main, no monkey-patching
needed here unlike vLLM's BlockSpaceManager.get_block_space_manager_class)."""

import dataclasses
from typing import Any, Callable, Dict, Optional

_BACKENDS: Dict[str, Callable[["TreeCacheBuildContext"], Any]] = {}


@dataclasses.dataclass
class TreeCacheBuildContext:
    server_args: Any = None
    params: Any = None
    is_hybrid_swa: bool = False
    is_hybrid_ssm: bool = False
    enable_hierarchical_cache: bool = False
    disable_radix_cache: bool = False
    model_config: Any = None


def register_radix_cache_backend(name: str, factory: Callable[[TreeCacheBuildContext], Any]) -> None:
    _BACKENDS[name] = factory


def get_radix_cache_factory(name: str) -> Optional[Callable[[TreeCacheBuildContext], Any]]:
    return _BACKENDS.get(name)


def registered_radix_cache_backends():
    return list(_BACKENDS.keys())


def create_tree_cache(ctx: TreeCacheBuildContext):
    raise NotImplementedError("fake registry stub: tests call the registered factory directly")
