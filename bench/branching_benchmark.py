"""
bench/branching_benchmark.py — Tree-of-Thought branching benchmark (Phase 3+).

This is the primary benchmark comparing AgentKV vs. stock PagedAttention /
RadixAttention on a branching workload.

STATUS: Phase 3 placeholder.  The framework integration benchmarks will be
implemented once vLLM / SGLang integrations are complete.

For now, this script runs the AgentKV-internal memory efficiency benchmark
(no framework integration required) to validate the CoW savings.

Metrics reported:
  - KV memory waste (% of blocks wasted due to duplication)
  - Peak memory usage (blocks)
  - Throughput simulation (tokens/sec, synthetic)

Usage:
  python bench/branching_benchmark.py --help
  python bench/branching_benchmark.py --mode internal --device cpu

Full framework comparison (Phase 3):
  python bench/branching_benchmark.py --mode vllm --model meta-llama/Llama-2-7b-hf
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import List

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from agentkv import AgentKVPool
from agentkv.core.config import PoolConfig


# ── Internal benchmark (no framework) ────────────────────────────────────────

def run_internal_benchmark(args) -> List[dict]:
    """Run the AgentKV-internal branching memory benchmark.

    This is a synthetic benchmark that does NOT require vLLM or SGLang.
    It measures:
      - AgentKV block usage for branching workloads
      - Equivalent naive usage (copy-per-branch)
      - Memory waste % and savings %
    """
    print("\n[AgentKV Internal Benchmark]")
    print(f"  device={args.device}, block_size={args.block_size}")
    print(f"  branching_factor={args.branching_factor}, depth={args.depth}")

    cfg = PoolConfig(
        total_blocks=args.total_blocks,
        block_size=args.block_size,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=args.dtype,
        device=args.device,
    )
    pool = AgentKVPool(config=cfg)

    results = []

    # Sweep multiple (branching_factor, depth) configs
    configs = [
        (2, 2), (2, 4), (4, 2), (4, 4), (8, 3),
    ]
    if (args.branching_factor, args.depth) not in configs:
        configs.append((args.branching_factor, args.depth))

    for bf, d in configs:
        try:
            result = _run_single_config(pool, bf, d, args)
            results.append(result)
            print(
                f"  B={bf:2d}, D={d}: "
                f"agents={result['n_agents']:5d}, "
                f"agentkv={result['agentkv_blocks']:6d} blks, "
                f"naive={result['naive_blocks']:6d} blks, "
                f"savings={result['savings_pct']:5.1f}%, "
                f"waste(naive)={result['waste_pct_naive']:5.1f}%"
            )
        except MemoryError:
            print(f"  B={bf}, D={d}: SKIPPED (pool too small)")

    return results


def _run_single_config(pool: AgentKVPool, bf: int, depth: int, args) -> dict:
    """Run one (branching_factor, depth) configuration."""
    prompt_blocks = args.prompt_blocks
    prompt_tokens = list(range(pool.block_size * prompt_blocks))

    root = pool.create_root(prompt_tokens)
    for _ in range(prompt_blocks):
        pool.allocate_block(root)

    all_handles = [root]
    current_layer = [root]

    for d in range(depth):
        next_layer = []
        for parent in current_layer:
            for _ in range(bf):
                child = pool.fork(parent)
                for _ in range(args.new_blocks_per_step):
                    pool.allocate_block(child)
                next_layer.append(child)
                all_handles.append(child)
        current_layer = next_layer

    agentkv_blocks = pool.allocated_blocks

    # Compute naive baseline
    naive = prompt_blocks  # root
    agents_at_d = 1
    cumulative_ancestor_blocks = prompt_blocks
    for d in range(1, depth + 1):
        agents_at_d *= bf
        per_agent = cumulative_ancestor_blocks + d * args.new_blocks_per_step
        naive += agents_at_d * per_agent
        cumulative_ancestor_blocks += args.new_blocks_per_step

    n_agents = len(all_handles)
    savings_pct = (1.0 - agentkv_blocks / max(naive, 1)) * 100
    # Memory waste = blocks that exist only due to duplication (naive - agentkv)
    waste_pct_naive = (naive - agentkv_blocks) / max(naive, 1) * 100

    for h in reversed(all_handles):
        pool.free(h)

    return {
        "branching_factor": bf,
        "depth": depth,
        "n_agents": n_agents,
        "agentkv_blocks": agentkv_blocks,
        "naive_blocks": naive,
        "savings_pct": savings_pct,
        "waste_pct_naive": waste_pct_naive,
        "total_pool_blocks": pool.config.total_blocks,
    }


# ── Framework benchmark (Phase 3) ─────────────────────────────────────────────

def run_vllm_benchmark(args):
    print("\n[vLLM Framework Benchmark] — Phase 3: NOT YET IMPLEMENTED")
    print("  This will compare AgentKV vs. stock vLLM PagedAttention on a")
    print("  branching workload with a real model.")
    print("  See integrations/vllm/README.md for the planned integration.")
    sys.exit(1)


def run_sglang_benchmark(args):
    print("\n[SGLang Framework Benchmark] — Phase 3: NOT YET IMPLEMENTED")
    sys.exit(1)


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(results: List[dict], output_dir: Path, ts: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"branching_benchmark_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved: {csv_path}")

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        labels = [f"B={r['branching_factor']}\nD={r['depth']}" for r in results]
        agentkv_v = [r["agentkv_blocks"] for r in results]
        naive_v = [r["naive_blocks"] for r in results]
        savings_v = [r["savings_pct"] for r in results]
        x = list(range(len(labels)))
        w = 0.35

        # Left: absolute blocks
        axes[0].bar([i - w / 2 for i in x], naive_v, w, label="Naive", color="#e74c3c", alpha=0.85)
        axes[0].bar([i + w / 2 for i in x], agentkv_v, w, label="AgentKV (CoW)", color="#2ecc71", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("KV Blocks Used")
        axes[0].set_title("Absolute Block Usage\n(lower = better)")
        axes[0].legend()

        # Right: savings %
        axes[1].bar(x, savings_v, color="#3498db", alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Memory Savings (%)")
        axes[1].set_title("CoW Memory Savings\n(higher = better)")
        axes[1].axhline(0, color="black", linewidth=0.8)

        plt.suptitle(
            "AgentKV Tree-of-Thought Branching Benchmark\n"
            "(Synthetic workload — no real model inference)",
            fontsize=11,
        )
        plt.tight_layout()
        plot_path = output_dir / f"branching_benchmark_{ts}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentKV Branching Benchmark")
    parser.add_argument("--mode", choices=["internal", "vllm", "sglang"], default="internal")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--total-blocks", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt-blocks", type=int, default=8,
                        help="Number of shared prompt blocks (tokens = prompt_blocks * block_size)")
    parser.add_argument("--new-blocks-per-step", type=int, default=2,
                        help="New blocks each agent allocates at each depth level")
    parser.add_argument("--output-dir", default="bench/results")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf",
                        help="Model for framework benchmarks (Phase 3)")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "internal":
        results = run_internal_benchmark(args)
        if results:
            save_results(results, Path(args.output_dir), ts)
    elif args.mode == "vllm":
        run_vllm_benchmark(args)
    elif args.mode == "sglang":
        run_sglang_benchmark(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
