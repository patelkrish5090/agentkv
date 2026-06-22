# AgentKV

> **⚠️ Status: Early Development (Pre-Alpha) — not production ready.**
> APIs will change without notice. Phase 0 (scaffolding) is complete; Phase 1 (core allocator) is in progress.

---

## What is AgentKV?

AgentKV is an open-source, GPU-resident KV-cache memory manager designed for **agentic LLM workloads** — specifically workloads involving:

- **Tree-of-Thought (ToT)** — branching reasoning trees where many agents share a long common prefix
- **ReAct / tool-use loops** — agents that fork new sub-agents mid-generation
- **Multi-agent branching** — many concurrent agents with heavily overlapping prompt prefixes

### The Problem

Standard KV cache managers (PagedAttention in vLLM, RadixAttention in SGLang) were designed for single-request batching. When an agentic workload *forks* — spawning 8 child agents from a common parent — each child currently gets its own full copy of the parent's KV blocks. For a 4K-token shared prefix with a 40 GB A100, this wastes gigabytes of VRAM that could serve more requests.

### The Solution

AgentKV replaces the allocator layer with a **Copy-on-Write (CoW) Radix Tree** that lives in GPU memory:

```
Parent agent: [prefix KV blocks — shared, read-only]
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
  Child A:      Child B:    Child C:
  [shared]      [shared]    [shared]
  [own Δ]       [own Δ]     [own Δ]
```

Children share the parent's KV blocks via reference counting. A block is only copied (the "Write" in CoW) when a child actually modifies it — which in autoregressive generation means only when a child generates its first *divergent* token. This gives sub-linear memory growth for branching workloads instead of linear.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Python API                           │
│  AgentKVPool  ·  fork()  ·  allocate_block()  ·  free()    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              Core Allocator  (Phase 1)                      │
│                                                             │
│  ┌──────────────────────────────┐  ┌──────────────────────┐ │
│  │    CoW Radix Tree           │  │   Slab Allocator     │ │
│  │  (CPU-resident metadata,   │  │  (GPU memory pool,   │ │
│  │   v1; GPU-resident in v2)  │  │   ref-counted)       │ │
│  └──────────────────────────────┘  └──────────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Triton Kernels                          │  │
│  │    copy_block · zero_block · match_prefix            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┐
        │                            │
┌───────▼────────┐          ┌────────▼───────┐
│  vLLM          │          │  SGLang        │
│  Integration   │          │  Integration   │
│  (Phase 3)     │          │  (Phase 3)     │
└────────────────┘          └────────────────┘
```

### Design Tradeoffs (v1)

| Decision | Choice | Rationale |
|---|---|---|
| Kernel authoring | Triton (not CUDA C++) | Ships as a pure Python wheel; no C++ toolchain required |
| Tree metadata residence | CPU (v1) | Far simpler correctness testing; GPU-resident tree is Phase 1b/v2 |
| Block granularity | Configurable (default: 16 tokens/block) | Matches vLLM's default PagedAttention block size |
| Reclamation | Epoch-based reference counting | Safe under concurrent reads; see `core/reclamation.py` for detailed rationale |
| Concurrency model | Thread-safe Python locks (v1); CUDA stream-ordered in kernels | CoW semantics require only read-side concurrency at the VRAM level |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/agentkv/agentkv
cd agentkv

# Install in editable mode (CPU-only, no Triton required)
pip install -e .

# Install with GPU kernel support (Linux/WSL2 only — Triton not on native Windows)
pip install -e ".[gpu]"

# Install with development tools
pip install -e ".[dev]"
```

> **Note on Windows**: Triton (the GPU kernel compiler) does not run on native Windows. If you're on Windows, use WSL2 for GPU-enabled development. The CPU-resident allocator logic (tree ops, Python tests) works on native Windows.

---

## Quickstart (Phase 2 — not yet implemented)

```python
from agentkv import AgentKVPool

# Allocate a 40 GB pool with 16-token KV blocks
pool = AgentKVPool(capacity_gb=40, block_size=16)

# Load a prompt into the root handle
root = pool.create_root(prompt_tokens=[1, 2, 3, ...])

# Fork to create child agents — they SHARE the parent's KV blocks
child_a = pool.fork(root)
child_b = pool.fork(root)
child_c = pool.fork(root)

# Each child allocates its own new blocks for divergent tokens
pool.allocate_block(child_a, n_tokens=16)

# Free when done — ref counting ensures shared blocks aren't freed early
pool.free(child_a)
pool.free(child_b)
pool.free(child_c)
pool.free(root)

print(pool.stats())
# → {'total_blocks': 512, 'allocated': 4, 'shared_savings_blocks': 6, ...}
```

> **⚠️ The above API is planned but not yet implemented.** Phase 0 (scaffolding) is the current milestone.

---

## Development Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Repo scaffolding, CI, packaging | ✅ Complete |
| **Phase 1** | CoW Radix Tree allocator + Triton kernels | 🔄 In Progress |
| **Phase 2** | Python bindings + `AgentKVPool` API | ⏳ Planned |
| **Phase 3** | vLLM + SGLang integrations | ⏳ Planned |
| **Phase 4** | Cooperative scheduling (research preview) | ⏳ Research preview |
| **Benchmarks** | ToT branching benchmark suite | ⏳ Planned |

---

## Running Tests

```bash
# CPU-only tests (no GPU required)
pytest tests/ -v -m "not gpu"

# All tests including GPU (requires CUDA-capable GPU)
AGENTKV_GPU_AVAILABLE=1 pytest tests/ -v

# Run the stress test
pytest tests/test_stress.py -v --timeout=300
```

---

## Performance Claims

> **⚠️ No benchmark numbers are presented here.** Performance claims will only appear in this README once a corresponding benchmark script in `bench/` has been committed and results have been reproduced on documented hardware. See `docs/BENCHMARKS.md` (coming in Phase 3+) for the honest record.

---

## What AgentKV Does NOT Do (v1)

- ❌ **GPU-initiated HTTPS calls to arbitrary external APIs** — TLS/HTTP termination from a CUDA kernel is not a solved problem. See Phase 4 notes on RDMA-only narrow networking.
- ❌ **Full cooperative kernel preemption** — requires DOCA GPUNetIO / RDMA-reachable services; see Phase 4.
- ❌ **Multi-node / distributed KV sharing** — single-node only in v1.
- ❌ **Dynamic cudaMalloc in the hot path** — all memory is pre-allocated from the pool at startup.

---

## Contributing

Contributions welcome! The project is in early development. Before contributing:

1. Read the [implementation plan](docs/architecture.md) to understand the design.
2. Open an issue to discuss major changes before implementing.
3. All PRs must include tests that fail on regression.
4. No performance claims without a committed benchmark script.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
