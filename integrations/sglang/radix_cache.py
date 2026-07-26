"""
AgentKVRadixCache — a real SGLang prefix-cache backend backed by AgentKV's
DualRadixTree, registered through SGLang's mem_cache backend registry.

Scope
-----
Unlike vLLM (where AgentKV replaces the whole BlockSpaceManager, including
physical CoW copies executed by vLLM's CacheEngine), SGLang's own
RadixAttention already performs the physical KV writes itself through
``token_to_kv_pool_allocator`` / ``req_to_token_pool`` — this adapter never
touches those. What it *does* own is the tree-shaped metadata SGLang's
scheduler needs for prefix sharing and eviction:

  - ``match_prefix``          — longest shared-prefix lookup for a new request
  - ``cache_unfinished_req``  — keep the tree in sync as a running request
                                 generates more tokens
  - ``cache_finished_req``    — promote a finished request's tokens into the
                                 shared tree (or drop them, if aborted)
  - ``evict`` / lock-ref       — eviction bookkeeping

There is no explicit ``fork()`` in SGLang's interface the way vLLM has one:
sharing between requests happens purely through repeated ``match_prefix``
calls against the same shared tree, which is exactly AgentKV's ``fork``
mechanism minus the RPC-style handle — see DualRadixTree.match_prefix.

Version note
------------
SGLang's mem_cache API (params-object style + the backend registry this
adapter plugs into) reflects SGLang `main` at commit ``10908a679`` (2026-07-18).
This is a fast-moving internal API — re-check signatures in
``sglang.srt.mem_cache.base_prefix_cache`` / ``registry`` before relying on
this against a specific pinned SGLang release.

Known limitations (v1)
-----------------------
- Page-level granularity only: a matched prefix is rounded down to whole
  pages, so up to one page's worth of an otherwise-matching prefix may be
  reported as a cache miss. Matches the same block-granularity tradeoff the
  vLLM integration makes.
- No hierarchical/host-memory cache, SWA, or mamba support — same v1 scope
  limitation as the rest of AgentKV (single-node, GPU-resident only).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.registry import TreeCacheBuildContext

from agentkv import AgentKVPool, PoolConfig
from agentkv.core.radix_tree import NodeHandle


class AgentKVRadixCache(BasePrefixCache):
    """Drop-in SGLang prefix-cache backend backed by AgentKV's DualRadixTree."""

    def __init__(self, params: Any) -> None:
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.disable = params.disable
        self._build_pool()

        # req.rid -> NodeHandle actively owned by a running/finished request.
        self._req_handles: Dict[str, NodeHandle] = {}
        # req.rid -> NodeHandle produced by match_prefix, awaiting either a
        # follow-up cache_unfinished_req/cache_finished_req (which adopts it
        # into _req_handles) or being discarded if the request never runs.
        self._match_handles: Dict[str, NodeHandle] = {}

    def _build_pool(self) -> None:
        # AgentKV's tree/ref-count bookkeeping lives on the "meta" device —
        # zero real VRAM — since SGLang's own token_to_kv_pool_allocator owns
        # the actual physical KV tensors.
        total_blocks = max(
            1, self.token_to_kv_pool_allocator.available_size() // self.page_size
        )
        cfg = PoolConfig(
            total_blocks=total_blocks,
            block_size=self.page_size,
            num_layers=1,
            num_kv_heads=1,
            head_dim=1,
            device="meta",
            dtype="float16",
        )
        self.pool = AgentKVPool(config=cfg)

    # ── BasePrefixCache ──────────────────────────────────────────────────────

    def reset(self) -> None:
        self._build_pool()
        self._req_handles.clear()
        self._match_handles.clear()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        tokens = list(params.key.token_ids)
        handle, match_len_raw = self.pool._tree.match_prefix(tokens)

        # Only whole pages are reportable as a hit — a matched handle whose
        # tail falls mid-page still holds the full raw match internally
        # (used to seed the request's own handle in _sync_req), we just
        # don't advertise the partial last page as reusable to the caller.
        match_len = (match_len_raw // self.page_size) * self.page_size

        if handle is None or match_len == 0:
            if handle is not None:
                self.pool._tree.free(handle)
            return MatchResult(device_indices=torch.empty(0, dtype=torch.int64))

        n_blocks = match_len // self.page_size
        block_ids = self.pool.get_block_ids(handle)[:n_blocks]
        device_indices = self._block_ids_to_indices(block_ids)

        rid = getattr(params.req, "rid", None)
        if rid is not None:
            self._match_handles[rid] = handle
        else:
            # No request context to adopt this handle later — release the
            # extra shared-node ref match_prefix took immediately.
            self.pool._tree.free(handle)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=handle,
            best_match_node=handle,
            full_kv_hit_length=match_len,
        )

    def cache_unfinished_req(self, req: Any, **kwargs: Any) -> None:
        self._sync_req(req, free_on_finish=False)

    def cache_finished_req(self, req: Any, is_insert: bool = True, **kwargs: Any) -> None:
        if is_insert:
            self._sync_req(req, free_on_finish=True)
            return
        handle = self._req_handles.pop(req.rid, None) or self._match_handles.pop(req.rid, None)
        if handle is not None:
            self.pool.free(handle)

    def evict(self, params: EvictParams) -> EvictResult:
        # Physical reclamation already happens automatically once ref counts
        # hit zero (AgentKV's epoch reclaimer) — there's no separate LRU walk
        # to perform here; just force an epoch advance so any handles freed
        # this scheduling step are reflected in available capacity now.
        before = self.pool.free_blocks
        self.pool.maybe_advance_epoch()
        after = self.pool.free_blocks
        return EvictResult(num_tokens_evicted=(after - before) * self.page_size)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        # The shared-tree ref count AgentKV already maintains (bumped by
        # match_prefix/fork, dropped by free) *is* the lock — nothing else
        # to pin here.
        return IncLockRefResult(delta=0)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        return DecLockRefResult(delta=0)

    def evictable_size(self) -> int:
        return self.pool.free_blocks * self.page_size

    def protected_size(self) -> int:
        return self.pool.allocated_blocks * self.page_size

    # ── internals ────────────────────────────────────────────────────────────

    def _block_ids_to_indices(self, block_ids) -> torch.Tensor:
        idx = []
        for b in block_ids:
            start = b * self.page_size
            idx.extend(range(start, start + self.page_size))
        return torch.tensor(idx, dtype=torch.int64)

    def _sync_req(self, req: Any, free_on_finish: bool) -> None:
        tokens = req.fill_ids() if hasattr(req, "fill_ids") else (
            list(req.origin_input_ids) + list(req.output_ids)
        )

        handle = self._req_handles.get(req.rid)
        if handle is None:
            handle = self._match_handles.pop(req.rid, None)
            if handle is None:
                handle = self.pool.create_root(tokens)
            self._req_handles[req.rid] = handle

        new_tokens = tokens[len(handle.tokens):]
        if new_tokens:
            self.pool.append_tokens(handle, new_tokens)

        required_blocks = (len(handle.tokens) + self.page_size - 1) // self.page_size
        while len(self.pool.get_block_ids(handle)) < required_blocks:
            self.pool.allocate_block(handle)

        committable = (
            (len(handle.tokens) // self.page_size) * self.page_size
            - handle.shared_match_len
        )
        if committable > 0:
            self.pool.commit_prefix(handle, committable)

        if free_on_finish:
            self._req_handles.pop(req.rid, None)
            self.pool.free(handle)


def _factory(ctx: TreeCacheBuildContext) -> AgentKVRadixCache:
    return AgentKVRadixCache(ctx.params)


def register() -> None:
    """Register AgentKVRadixCache as an SGLang prefix-cache backend.

    Call this before launching the SGLang server, then pass
    ``--radix-cache-backend agentkv`` on the command line (or the
    equivalent server_args field when embedding SGLang directly).
    """
    try:
        from sglang.srt.mem_cache.registry import register_radix_cache_backend
    except ImportError as e:
        raise ImportError(
            "SGLang is not installed. Please install sglang to use the "
            "AgentKV SGLang integration."
        ) from e

    register_radix_cache_backend("agentkv", _factory)


__all__ = ["AgentKVRadixCache", "register"]
