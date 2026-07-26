"""
tests/test_sglang_integration.py — Tests for AgentKVRadixCache (integrations/sglang).

Real SGLang cannot be installed in this dev environment (CUDA build
required). These tests run the adapter against a minimal stub of
sglang.srt.mem_cache in tests/fakes/, whose signatures were copied from
SGLang's `main` branch (see integrations/sglang/README.md for the version
pin). This validates AgentKVRadixCache's actual logic — prefix matching,
request lifecycle, eviction bookkeeping — without a real SGLang install or
a GPU.
"""

import os
import sys

import pytest

_FAKES_DIR = os.path.join(os.path.dirname(__file__), "fakes")
if _FAKES_DIR not in sys.path:
    sys.path.insert(0, _FAKES_DIR)

from sglang.srt.mem_cache.allocator.base import FakePagedAllocator  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    MatchPrefixParams,
    RadixKey,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402

from integrations.sglang.radix_cache import AgentKVRadixCache  # noqa: E402

pytestmark = pytest.mark.integration

PAGE_SIZE = 16
NUM_TOKEN_SLOTS = 256 * PAGE_SIZE


@pytest.fixture
def cache() -> AgentKVRadixCache:
    params = CacheInitParams(
        req_to_token_pool=ReqToTokenPool(size=256, max_context_len=4096),
        token_to_kv_pool_allocator=FakePagedAllocator(size=NUM_TOKEN_SLOTS, page_size=PAGE_SIZE),
        page_size=PAGE_SIZE,
    )
    return AgentKVRadixCache(params)


def test_match_prefix_no_hit_on_empty_cache(cache: AgentKVRadixCache):
    req = Req("r1", list(range(32)))
    result = cache.match_prefix(MatchPrefixParams(key=RadixKey(req.origin_input_ids), req=req))
    assert result.full_kv_hit_length == 0
    assert result.device_indices.numel() == 0


def test_concurrent_request_reuses_committed_prefix(cache: AgentKVRadixCache):
    """A committed prefix is reusable by other requests *while at least one
    referencing handle is still alive* — this is AgentKV's core sharing
    model (concurrently-active agents, e.g. branching ToT/multi-agent), not
    a time-persistent cache: once every referencing request finishes, the
    shared node is pruned immediately (see AgentKVRadixCache README)."""
    req_a = Req("r1", list(range(32)))  # 2 full pages
    match_a = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req_a.origin_input_ids), req=req_a)
    )
    assert match_a.full_kv_hit_length == 0
    cache.cache_unfinished_req(req_a)  # req_a stays alive, keeping the ref

    # Second, concurrent request shares the same 32-token prefix plus a new
    # page's worth of its own tokens.
    req_b = Req("r2", list(range(32)) + list(range(1000, 1016)))
    match_b = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req_b.origin_input_ids), req=req_b)
    )
    assert match_b.full_kv_hit_length == 32
    assert match_b.device_indices.numel() == 32

    cache.cache_finished_req(req_b, is_insert=True)
    cache.cache_finished_req(req_a, is_insert=True)


def test_unfinished_request_syncs_incrementally(cache: AgentKVRadixCache):
    req = Req("r1", list(range(16)))
    cache.match_prefix(MatchPrefixParams(key=RadixKey(req.origin_input_ids), req=req))
    cache.cache_unfinished_req(req)

    # Simulate one decode step generating a new token.
    req.output_ids.append(9001)
    cache.cache_unfinished_req(req)

    handle = cache._req_handles[req.rid]
    assert len(handle.tokens) == 17

    cache.cache_finished_req(req, is_insert=True)
    assert req.rid not in cache._req_handles


def test_aborted_request_is_discarded_not_committed(cache: AgentKVRadixCache):
    req_a = Req("r1", list(range(16)))
    cache.cache_unfinished_req(req_a)
    cache.cache_finished_req(req_a, is_insert=False)  # aborted, don't commit

    req_b = Req("r2", list(range(16)))
    match_b = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(req_b.origin_input_ids), req=req_b)
    )
    assert match_b.full_kv_hit_length == 0  # nothing was committed to share


def test_evict_reports_reclaimed_capacity(cache: AgentKVRadixCache):
    req = Req("r1", list(range(16)))
    cache.cache_finished_req(req, is_insert=True)

    free_before = cache.pool.free_blocks
    result = cache.evict(EvictParams(num_tokens=PAGE_SIZE))
    free_after = cache.pool.free_blocks

    assert free_after >= free_before
    assert result.num_tokens_evicted == (free_after - free_before) * PAGE_SIZE


def test_reset_clears_all_state(cache: AgentKVRadixCache):
    req = Req("r1", list(range(16)))
    cache.cache_unfinished_req(req)  # keep it alive so its blocks aren't pruned
    assert cache.pool.allocated_blocks > 0 or cache.pool.stats()["shared_blocks"] > 0

    cache.reset()
    assert cache._req_handles == {}
    assert cache._match_handles == {}
    assert cache.pool.allocated_blocks == 0
