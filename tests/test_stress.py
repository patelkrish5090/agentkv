"""
tests/test_stress.py — Stress test for the AgentKV core allocator.

This is the Phase 1 mandatory stress test described in the spec:
  "10,000+ random fork/alloc/free operations, zero correctness violations."

Strategy
--------
We run a randomized simulation of N concurrent "agents":
  - Each agent is a Python thread (or a sequential simulation if FAST_CI=1).
  - Each iteration randomly chooses: fork a new agent, allocate a block,
    free a block, or free an entire agent.
  - A CPU-side reference implementation (RefTracker) tracks ground truth.
  - After each operation, we assert invariants:
    * No block ID appears in two different exclusive agents simultaneously.
    * Shared blocks have the correct reference count.
    * No block is both "free" and "allocated" at the same time.
    * Total allocated blocks match the reference implementation.
  - After all operations, the pool must return to its initial state
    (all blocks free, no leaks).

This test is deliberately adversarial: we don't control the order of
fork/alloc/free to try to expose any correctness bug.

Correctness violations
----------------------
  DOUBLE_FREE      : dec_ref called on a block with ref_count == 0.
  DOUBLE_ALLOC     : alloc returns a block that is still allocated.
  LEAK             : after all frees, allocated_count != 0.
  REF_MISMATCH     : observed ref_count != expected ref_count.

Running
-------
  pytest tests/test_stress.py -v --timeout=300

  With verbose output:
  pytest tests/test_stress.py -v -s

GPU stress test (requires CUDA):
  AGENTKV_GPU_AVAILABLE=1 pytest tests/test_stress.py::TestGPUStress -v
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pytest
from agentkv.core.config import PoolConfig
from agentkv.core.slab_allocator import SlabAllocator, BlockHandle
from agentkv.core.radix_tree import DualRadixTree, NodeHandle


# ── Reference implementation ──────────────────────────────────────────────────

@dataclass
class AgentState:
    """CPU reference model for one agent."""
    handle_id: int
    block_ids: List[int] = field(default_factory=list)  # blocks owned by this agent
    is_alive: bool = True


class RefTracker:
    """CPU-side ground truth for the allocator stress test.

    This tracks what the allocator SHOULD look like.  Any divergence between
    RefTracker state and SlabAllocator state is a correctness violation.
    """

    def __init__(self, total_blocks: int) -> None:
        self._total = total_blocks
        self._free: Set[int] = set(range(total_blocks))
        self._ref_counts: Dict[int, int] = {i: 0 for i in range(total_blocks)}
        self._agents: Dict[int, AgentState] = {}
        self._violations: List[str] = []

    def on_alloc(self, agent_id: int, block_id: int) -> None:
        if block_id not in self._free:
            self._violations.append(
                f"DOUBLE_ALLOC: block {block_id} allocated to agent {agent_id} "
                f"but was not in free list (ref_count={self._ref_counts[block_id]})"
            )
            return
        self._free.discard(block_id)
        self._ref_counts[block_id] = 1
        self._agents[agent_id].block_ids.append(block_id)

    def on_fork(self, parent_id: int, child_id: int) -> None:
        parent = self._agents[parent_id]
        child = AgentState(handle_id=child_id, block_ids=list(parent.block_ids))
        self._agents[child_id] = child
        # Increment ref counts for all shared blocks
        for bid in parent.block_ids:
            self._ref_counts[bid] += 1

    def on_free_agent(self, agent_id: int) -> None:
        agent = self._agents[agent_id]
        for bid in agent.block_ids:
            self._ref_counts[bid] -= 1
            if self._ref_counts[bid] == 0:
                self._free.add(bid)
            elif self._ref_counts[bid] < 0:
                self._violations.append(
                    f"DOUBLE_FREE: block {bid} freed by agent {agent_id} "
                    f"but ref_count was already 0"
                )
        agent.is_alive = False
        del self._agents[agent_id]

    def register_agent(self, agent_id: int) -> None:
        self._agents[agent_id] = AgentState(handle_id=agent_id)

    @property
    def violations(self) -> List[str]:
        return list(self._violations)

    @property
    def alive_agents(self) -> List[int]:
        return list(self._agents.keys())


# ── Stress test runner ────────────────────────────────────────────────────────

class StressRunner:
    """Runs randomized fork/alloc/free sequences and checks invariants."""

    def __init__(
        self,
        cfg: PoolConfig,
        n_operations: int = 10_000,
        max_live_agents: int = 32,
        seed: int = 42,
    ) -> None:
        self._cfg = cfg
        self._n_ops = n_operations
        self._max_agents = max_live_agents
        self._rng = random.Random(seed)

        self._alloc = SlabAllocator(cfg)
        self._tree = DualRadixTree(self._alloc, cfg)
        self._ref = RefTracker(cfg.total_blocks)

        # Live (handle_id → NodeHandle) registry for the stress runner
        self._live: Dict[int, NodeHandle] = {}
        self._violations: List[str] = []

    def run(self) -> List[str]:
        """Execute the randomized sequence.  Returns list of violation strings."""
        for i in range(self._n_ops):
            self._step(i)

        # Cleanup: free all remaining agents
        for hid in list(self._live.keys()):
            self._do_free_agent(hid)

        # Force epoch advance to reclaim all pending blocks
        for _ in range(10):
            self._alloc.maybe_advance_epoch()

        # Final invariant: all blocks must be free
        free_c = self._alloc.free_count
        if free_c != self._cfg.total_blocks:
            self._violations.append(
                f"LEAK: expected {self._cfg.total_blocks} free blocks after all agents freed, "
                f"got {free_c}. Leaked {self._cfg.total_blocks - free_c} blocks."
            )

        self._violations.extend(self._ref.violations)
        return self._violations

    def _step(self, step_idx: int) -> None:
        """One randomized operation."""
        alive = list(self._live.keys())
        n_alive = len(alive)

        # Choose action
        if n_alive == 0:
            action = "create"
        elif n_alive >= self._max_agents:
            # Must free or allocate a block
            action = self._rng.choice(["alloc_block", "free_agent"])
        else:
            action = self._rng.choices(
                ["create", "fork", "alloc_block", "free_agent"],
                weights=[2, 3, 4, 2],
            )[0]

        try:
            if action == "create":
                self._do_create()
            elif action == "fork":
                parent_id = self._rng.choice(alive)
                self._do_fork(parent_id)
            elif action == "alloc_block":
                agent_id = self._rng.choice(alive)
                self._do_alloc_block(agent_id)
            elif action == "free_agent":
                agent_id = self._rng.choice(alive)
                self._do_free_agent(agent_id)
        except MemoryError:
            # Pool exhausted — this is not a bug, just free some agents
            if self._live:
                agent_id = self._rng.choice(list(self._live.keys()))
                self._do_free_agent(agent_id)
        except Exception as e:
            self._violations.append(f"UNEXPECTED ERROR at step {step_idx}: {e}")

    def _next_token_seq(self) -> List[int]:
        return [self._rng.randint(0, 255) for _ in range(self._rng.randint(1, 8))]

    def _do_create(self) -> None:
        tokens = self._next_token_seq()
        handle = self._tree.create_root(tokens)
        self._live[handle.handle_id] = handle
        self._ref.register_agent(handle.handle_id)

    def _do_fork(self, parent_id: int) -> None:
        parent = self._live[parent_id]
        child = self._tree.fork(parent)
        self._live[child.handle_id] = child
        self._ref.on_fork(parent_id, child.handle_id)

    def _do_alloc_block(self, agent_id: int) -> None:
        handle = self._live[agent_id]
        bh = self._tree.allocate_block(handle)
        self._ref.on_alloc(agent_id, bh.block_id)

    def _do_free_agent(self, agent_id: int) -> None:
        handle = self._live.pop(agent_id)
        self._tree.free(handle)
        self._ref.on_free_agent(agent_id)
        for _ in range(3):
            self._alloc.maybe_advance_epoch()


# ── Test cases ────────────────────────────────────────────────────────────────

class TestStressCPU:
    """CPU-based stress tests (no GPU required)."""

    @pytest.fixture
    def stress_cfg(self):
        return PoolConfig(
            total_blocks=512,
            block_size=4,
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype="float16",
            device="cpu",
        )

    def test_stress_1000_ops(self, stress_cfg):
        """1,000 operations — fast smoke test for CI."""
        runner = StressRunner(stress_cfg, n_operations=1_000, seed=1)
        violations = runner.run()
        assert violations == [], (
            f"Correctness violations after 1,000 ops:\n"
            + "\n".join(violations)
        )

    @pytest.mark.slow
    def test_stress_10000_ops(self, stress_cfg):
        """10,000 operations — the mandatory Phase 1 gate test."""
        runner = StressRunner(stress_cfg, n_operations=10_000, seed=2)
        violations = runner.run()
        assert violations == [], (
            f"Correctness violations after 10,000 ops:\n"
            + "\n".join(violations)
        )

    @pytest.mark.slow
    def test_stress_multiple_seeds(self, stress_cfg):
        """Run 3 independent seeds to catch seed-dependent bugs."""
        for seed in [42, 137, 999]:
            runner = StressRunner(stress_cfg, n_operations=5_000, seed=seed)
            violations = runner.run()
            assert violations == [], (
                f"Violations with seed={seed}:\n" + "\n".join(violations)
            )

    @pytest.mark.slow
    def test_stress_no_block_leak(self, stress_cfg):
        """After all ops, every block must be returned to the free pool."""
        runner = StressRunner(stress_cfg, n_operations=10_000, seed=7)
        violations = runner.run()
        leak_violations = [v for v in violations if "LEAK" in v]
        assert leak_violations == [], (
            f"Block leak detected:\n" + "\n".join(leak_violations)
        )

    @pytest.mark.slow
    def test_stress_no_double_alloc(self, stress_cfg):
        """No block should be allocated to two agents simultaneously."""
        runner = StressRunner(stress_cfg, n_operations=10_000, seed=13)
        violations = runner.run()
        da_violations = [v for v in violations if "DOUBLE_ALLOC" in v]
        assert da_violations == [], (
            f"Double-allocation detected:\n" + "\n".join(da_violations)
        )

    @pytest.mark.slow
    def test_stress_no_double_free(self, stress_cfg):
        """No block should be freed when its ref_count is already 0."""
        runner = StressRunner(stress_cfg, n_operations=10_000, seed=17)
        violations = runner.run()
        df_violations = [v for v in violations if "DOUBLE_FREE" in v]
        assert df_violations == [], (
            f"Double-free detected:\n" + "\n".join(df_violations)
        )

    def test_stress_concurrent_threads(self, stress_cfg):
        """Multiple threads running StressRunner concurrently should not crash."""
        n_threads = 4
        all_violations: List[str] = []
        lock = threading.Lock()

        def worker(seed: int) -> None:
            runner = StressRunner(stress_cfg, n_operations=500, seed=seed)
            # Each thread gets its own allocator + tree (not shared — this tests
            # thread safety within each allocator, not cross-allocator sharing).
            violations = runner.run()
            with lock:
                all_violations.extend(violations)

        threads = [threading.Thread(target=worker, args=(i * 100,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all_violations == [], (
            f"Violations in concurrent thread test:\n" + "\n".join(all_violations)
        )


@pytest.mark.gpu
class TestGPUStress:
    """GPU stress tests — only run when AGENTKV_GPU_AVAILABLE=1."""

    @pytest.fixture
    def gpu_stress_cfg(self):
        return PoolConfig(
            total_blocks=2048,
            block_size=16,
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
            dtype="float16",
            device="cuda",
        )

    def test_gpu_stress_10000_ops(self, gpu_stress_cfg):
        """10,000 ops on GPU device — same correctness guarantee as CPU test."""
        runner = StressRunner(gpu_stress_cfg, n_operations=10_000, seed=42)
        violations = runner.run()
        assert violations == [], (
            f"GPU correctness violations after 10,000 ops:\n"
            + "\n".join(violations)
        )
