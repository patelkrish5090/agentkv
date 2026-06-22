"""
bench/microbench_alloc.py — Allocation latency and memory overhead microbenchmark.

Measures:
  1. Allocation latency (time per alloc, inc_ref, dec_ref)
  2. Memory efficiency: AgentKV CoW sharing vs naive copy-per-branch baseline

Workload: simulates Tree-of-Thought with branching factor B, depth D.
  - A root agent is created with a long shared prompt.
  - At each depth level, each live agent forks B children.
  - Total agents: 1 + B + B^2 + ... + B^D

Usage:
  python bench/microbench_alloc.py
  python bench/microbench_alloc.py --branching-factor 4 --depth 6 --device cpu

Output:
  - CSV: bench/results/microbench_TIMESTAMP.csv
  - Plot: bench/results/microbench_TIMESTAMP.png
  - Console summary
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from agentkv import AgentKVPool
from agentkv.core.config import PoolConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_tot_tree(
    pool: AgentKVPool,
    branching_factor: int,
    depth: int,
    prompt_len: int,
    blocks_per_agent: int,
) -> Tuple[List, int]:
    """Build a Tree-of-Thought agent tree.

    Returns (all_handles, total_blocks_allocated).
    """
    # Root
    prompt_tokens = list(range(prompt_len))
    root = pool.create_root(prompt_tokens)

    # Give root some prompt blocks
    for _ in range(blocks_per_agent):
        pool.allocate_block(root)

    all_handles = [root]
    current_layer = [root]

    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            for _ in range(branching_factor):
                child = pool.fork(parent)
                # Each child allocates a few unique blocks (divergent tokens)
                for _ in range(2):
                    pool.allocate_block(child)
                next_layer.append(child)
                all_handles.append(child)
        current_layer = next_layer

    return all_handles, pool.allocated_blocks


def naive_baseline_blocks(
    branching_factor: int,
    depth: int,
    prompt_blocks: int,
    blocks_per_agent: int,
) -> int:
    """Compute how many blocks a naive copy-per-branch baseline would need.

    In the naive scheme, each agent gets its own copy of all ancestor blocks.
    So total blocks = sum over each agent of (its_depth * prompt_blocks + own_blocks)
    """
    total = prompt_blocks  # root
    agents_at_depth = 1
    cumulative_shared_per_agent = prompt_blocks

    for d in range(1, depth + 1):
        agents_at_depth *= branching_factor
        # Each agent at this depth: copies of all parent blocks + its own
        blocks_per_agent_naive = cumulative_shared_per_agent + (d * blocks_per_agent * 2)
        total += agents_at_depth * blocks_per_agent_naive

    return total


# ── Benchmarks ────────────────────────────────────────────────────────────────

def bench_alloc_latency(pool: AgentKVPool, n_iter: int = 10_000) -> Dict:
    """Measure raw alloc / inc_ref / dec_ref latency."""
    handle = pool.create_root([])

    # Warm up
    for _ in range(100):
        pool.allocate_block(handle)

    # Clear
    pool.free(handle)
    handle = pool.create_root([])

    # Alloc
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pool.allocate_block(handle)
    alloc_time = (time.perf_counter() - t0) / n_iter * 1e6  # µs per op

    # Free all (clean up for next bench)
    pool.free(handle)
    handle = pool.create_root([])

    # Fork + free (measures inc_ref path)
    t0 = time.perf_counter()
    for _ in range(1000):
        child = pool.fork(handle)
        pool.free(child)
    fork_time = (time.perf_counter() - t0) / 1000 * 1e6

    pool.free(handle)

    return {
        "alloc_us": alloc_time,
        "fork_us": fork_time,
        "n_iter": n_iter,
    }


def bench_memory_efficiency(
    pool: AgentKVPool,
    branching_factor: int,
    depth: int,
    prompt_blocks: int = 16,
    blocks_per_agent: int = 2,
) -> Dict:
    """Measure memory savings from CoW sharing vs. naive baseline."""
    handles, agentkv_blocks = build_tot_tree(
        pool,
        branching_factor=branching_factor,
        depth=depth,
        prompt_len=pool.block_size * prompt_blocks,
        blocks_per_agent=blocks_per_agent,
    )

    naive_blocks = naive_baseline_blocks(
        branching_factor=branching_factor,
        depth=depth,
        prompt_blocks=prompt_blocks,
        blocks_per_agent=blocks_per_agent,
    )

    n_agents = len(handles)
    savings_pct = (1 - agentkv_blocks / max(naive_blocks, 1)) * 100

    # Cleanup
    for h in reversed(handles):
        pool.free(h)

    return {
        "branching_factor": branching_factor,
        "depth": depth,
        "n_agents": n_agents,
        "agentkv_blocks": agentkv_blocks,
        "naive_blocks": naive_blocks,
        "savings_pct": savings_pct,
        "total_blocks": pool.config.total_blocks,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentKV Microbenchmark")
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-blocks", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"AgentKV Microbenchmark")
    print(f"  device={args.device}, B={args.branching_factor}, D={args.depth}")
    print(f"  total_blocks={args.total_blocks}, block_size={args.block_size}")
    print(f"{'='*60}\n")

    cfg = PoolConfig(
        total_blocks=args.total_blocks,
        block_size=args.block_size,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
        dtype="float16",
        device=args.device,
    )
    pool = AgentKVPool(config=cfg)

    # Latency benchmark
    print("1. Allocation latency...")
    lat = bench_alloc_latency(pool, n_iter=min(5000, args.total_blocks - 100))
    print(f"   alloc: {lat['alloc_us']:.2f} µs/op")
    print(f"   fork:  {lat['fork_us']:.2f} µs/op")

    # Memory efficiency benchmark across multiple (B, D) configs
    print("\n2. Memory efficiency (CoW vs naive)...")
    results = []
    configs = [
        (2, 3), (4, 3), (4, 4), (8, 3),
        (args.branching_factor, args.depth),
    ]
    seen = set()
    for b, d in configs:
        key = (b, d)
        if key in seen:
            continue
        seen.add(key)
        try:
            r = bench_memory_efficiency(pool, branching_factor=b, depth=d)
            results.append(r)
            print(
                f"   B={b}, D={d}: {r['n_agents']} agents, "
                f"AgentKV={r['agentkv_blocks']} blocks, "
                f"naive={r['naive_blocks']} blocks, "
                f"savings={r['savings_pct']:.1f}%"
            )
        except MemoryError:
            print(f"   B={b}, D={d}: pool too small for this configuration")

    # Save results
    results_dir = Path("bench/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV
    csv_path = results_dir / f"microbench_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\n   Results saved to {csv_path}")

    # Plot
    if HAS_MATPLOTLIB and results:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"B={r['branching_factor']},D={r['depth']}" for r in results]
        agentkv_vals = [r["agentkv_blocks"] for r in results]
        naive_vals = [r["naive_blocks"] for r in results]
        x = range(len(labels))
        width = 0.35
        ax.bar([i - width / 2 for i in x], naive_vals, width, label="Naive (copy-per-branch)", color="#e74c3c", alpha=0.85)
        ax.bar([i + width / 2 for i in x], agentkv_vals, width, label="AgentKV (CoW sharing)", color="#2ecc71", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel("KV Blocks Used")
        ax.set_title("AgentKV CoW Sharing vs Naive Copy-Per-Branch\n(lower = better)")
        ax.legend()
        plt.tight_layout()
        plot_path = results_dir / f"microbench_{ts}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"   Plot saved to {plot_path}")
    elif not HAS_MATPLOTLIB:
        print("   (matplotlib not installed — skipping plot)")

    print(f"\n{'='*60}")
    print("Benchmark complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
