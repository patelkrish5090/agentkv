"""
tests/conftest.py — Shared pytest fixtures and configuration.

Markers:
    gpu         — requires CUDA-capable GPU (skip if AGENTKV_GPU_AVAILABLE != 1)
    slow        — takes > 10s; skipped in fast CI mode
    integration — requires installed vLLM / SGLang

Environment variables:
    AGENTKV_GPU_AVAILABLE=1   — enables GPU tests
    AGENTKV_FAST_CI=1         — skips slow tests
"""

import os
import pytest


# ── GPU skip decorator ────────────────────────────────────────────────────────
GPU_AVAILABLE = os.environ.get("AGENTKV_GPU_AVAILABLE", "0") == "1"
FAST_CI = os.environ.get("AGENTKV_FAST_CI", "0") == "1"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "gpu: marks tests requiring a CUDA-capable GPU (use AGENTKV_GPU_AVAILABLE=1 to enable)",
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests that take > 10 seconds",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests requiring installed vLLM or SGLang",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "gpu" in item.keywords and not GPU_AVAILABLE:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "GPU test skipped — set AGENTKV_GPU_AVAILABLE=1 to run. "
                        "Requires a CUDA-capable GPU and Triton installed."
                    )
                )
            )
        if "slow" in item.keywords and FAST_CI:
            item.add_marker(
                pytest.mark.skip(reason="Slow test skipped — AGENTKV_FAST_CI=1 set")
            )


# ── Common fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def small_pool_config():
    """A small pool config suitable for CPU-only unit tests."""
    from agentkv.core.config import PoolConfig
    return PoolConfig(
        total_blocks=256,
        block_size=16,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        dtype="float16",
        device="cpu",
    )


@pytest.fixture
def medium_pool_config():
    """A medium pool config for stress tests."""
    from agentkv.core.config import PoolConfig
    return PoolConfig(
        total_blocks=4096,
        block_size=16,
        num_layers=4,
        num_kv_heads=8,
        head_dim=128,
        dtype="float16",
        device="cpu",
    )


@pytest.fixture
def gpu_pool_config():
    """A GPU pool config for on-device tests (requires CUDA)."""
    from agentkv.core.config import PoolConfig
    return PoolConfig(
        total_blocks=8192,
        block_size=16,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype="float16",
        device="cuda",
    )
