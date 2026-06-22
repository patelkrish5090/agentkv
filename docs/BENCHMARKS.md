# Benchmarks

> **Status: No benchmark results yet.**

Benchmark results will be published here in Phase 3+ with full reproducibility details:
- Exact model name + HuggingFace revision hash
- GPU model (RTX 5000 Ada, sm_89) + driver version + CUDA version
- Workload: Tree-of-Thought branching factor, depth, prompt lengths
- Metrics: KV memory waste (%), peak memory (GB), throughput (tokens/sec)

See `bench/branching_benchmark.py` for the benchmark script (Phase 3).
