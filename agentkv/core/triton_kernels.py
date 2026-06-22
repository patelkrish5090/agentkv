"""
agentkv/core/triton_kernels.py — Triton device kernels for KV block operations.

Supports any GPU that Triton targets (sm_70+). The tile size is auto-tuned
per GPU architecture at first JIT compile. Tested on:
  - T4  (sm_75, Turing)       — Google Colab free tier
  - A10 (sm_86, Ampere)
  - A100 (sm_80, Ampere)
  - RTX 5000 Ada (sm_89, Ada Lovelace)

Architecture-specific notes
----------------------------
  T4 (sm_75 / Turing):
    - 32 B L1 cache lines  →  BLOCK_SIZE=256 fp16 elements = 512 B (good fit)
    - bfloat16 NOT natively supported on Turing; the kernels guard against this
      and fall back to torch.Tensor ops if bfloat16 is requested on a Turing GPU.
    - FP16 throughput is excellent on T4 (65 TFLOPS FP16).

  RTX 5000 Ada / Ada Lovelace (sm_89):
    - 128 B L1 cache lines  →  BLOCK_SIZE=512 fp16 elements = 1 KB (good fit)

Tile size selection
-------------------
We use Triton's @triton.autotune to select BLOCK_SIZE at first kernel launch.
This means the first call has a one-time JIT compilation cost (a few seconds),
but subsequent calls are fast. The compiled binary is cached in ~/.triton/cache.

GPU availability detection
--------------------------
If Triton is not installed (native Windows, CPU-only CI), all functions fall
back to torch.Tensor ops.  No GPU is required for CPU-only testing.
"""

from __future__ import annotations

import os
import torch
from typing import Optional

# ── Triton availability guard ─────────────────────────────────────────────────
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ── GPU architecture detection ────────────────────────────────────────────────

def _detect_gpu_arch() -> str:
    """Return a human-readable GPU architecture string."""
    if not torch.cuda.is_available():
        return "cpu"
    cap = torch.cuda.get_device_capability()
    major, minor = cap
    sm = major * 10 + minor
    arch_map = {
        70: "sm_70 (Volta, e.g. V100)",
        75: "sm_75 (Turing, e.g. T4/RTX 20xx)",
        80: "sm_80 (Ampere, e.g. A100)",
        86: "sm_86 (Ampere, e.g. A10/RTX 30xx)",
        89: "sm_89 (Ada Lovelace, e.g. RTX 40xx/RTX 5000 Ada)",
        90: "sm_90 (Hopper, e.g. H100)",
    }
    return arch_map.get(sm, f"sm_{sm} (unknown)")


def _get_optimal_block_size() -> int:
    """Return the optimal Triton tile size for the detected GPU.

    Heuristic based on L1 cache line width:
      Turing (T4):   32 B lines  → 256 fp16 elements per tile
      Ampere/Ada:   128 B lines  → 512 fp16 elements per tile
      Hopper:       128 B lines  → 512 fp16 elements per tile
      CPU / unknown:             → 256 (conservative)
    """
    if not torch.cuda.is_available():
        return 256
    major, _ = torch.cuda.get_device_capability()
    # Turing = sm_7x: smaller cache lines → smaller tile
    if major == 7:
        return 256
    # Ampere, Ada, Hopper = sm_8x, sm_9x: larger cache lines → bigger tile
    return 512


def _supports_bfloat16_gpu() -> bool:
    """Returns True if the current GPU natively supports bfloat16.

    Turing (sm_75) does NOT support bfloat16; Ampere+ does.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8  # Ampere (sm_80) and above


# ── Triton kernel definitions (only compiled when Triton is available) ─────────
# NOTE: Triton's @triton.jit compiles at first call, not at import time.
# The BLOCK_SIZE is a constexpr so the compiler can unroll loops.
# Autotune selects the best config per GPU at first run.

if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _copy_block_kernel(
        src_ptr,
        dst_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Copy n_elements fp16/bf16 values from src_ptr → dst_ptr.

        Autotuned: Triton picks the best BLOCK_SIZE for the detected GPU
        at first launch (cached afterward in ~/.triton/cache).
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, data, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}),
            triton.Config({"BLOCK_SIZE": 256}),
            triton.Config({"BLOCK_SIZE": 512}),
            triton.Config({"BLOCK_SIZE": 1024}),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _zero_block_kernel(
        dst_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Zero-fill n_elements values at dst_ptr.

        NOTE: We use tl.load with a zero mask to infer the dtype from the
        pointer, then store zeros of the same type.  This avoids the bug in
        the original implementation which hardcoded tl.float16 regardless of
        the actual tensor dtype.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        # Load to get the dtype of the pointer, multiply by 0 to zero it.
        # This is a portable zero that works for fp16, fp32, bf16.
        data = tl.load(dst_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, data * 0, mask=mask)

else:
    _copy_block_kernel = None  # type: ignore[assignment]
    _zero_block_kernel = None  # type: ignore[assignment]


# ── Public dispatch functions ─────────────────────────────────────────────────

def copy_block(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy a KV block tensor from src to dst.

    Parameters
    ----------
    src : torch.Tensor
        Source block.  Any dtype/device.
    dst : torch.Tensor
        Destination block.  Same shape as src.
        Caller guarantees exclusive ownership (ref_count == 1).
    """
    assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"

    use_triton = (
        _TRITON_AVAILABLE
        and src.device.type == "cuda"
        and src.is_contiguous()
        and dst.is_contiguous()
        and src.dtype in (torch.float16, torch.bfloat16, torch.float32)
        # bfloat16 guard: don't use Triton on Turing (sm_75) with bf16
        and not (src.dtype == torch.bfloat16 and not _supports_bfloat16_gpu())
    )

    if not use_triton:
        dst.copy_(src)
        return

    # GPU path: Triton kernel (autotuned)
    src_flat = src.view(-1).contiguous()
    dst_flat = dst.view(-1)
    n = src_flat.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _copy_block_kernel[grid](src_flat, dst_flat, n)


def zero_block(block: torch.Tensor) -> None:
    """Zero-fill a KV block tensor.

    Parameters
    ----------
    block : torch.Tensor
        Block to zero.  Caller guarantees exclusive ownership.
    """
    use_triton = (
        _TRITON_AVAILABLE
        and block.device.type == "cuda"
        and block.is_contiguous()
        and block.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and not (block.dtype == torch.bfloat16 and not _supports_bfloat16_gpu())
    )

    if not use_triton:
        block.zero_()
        return

    flat = block.view(-1).contiguous()
    n = flat.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _zero_block_kernel[grid](flat, n)


def is_triton_available() -> bool:
    """Returns True if Triton is installed and a CUDA GPU is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


def kernel_info() -> dict:
    """Return information about the current GPU and kernel configuration."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    return {
        "triton_available": _TRITON_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "gpu_arch": _detect_gpu_arch(),
        "bfloat16_supported": _supports_bfloat16_gpu(),
        "optimal_block_size": _get_optimal_block_size(),
        "fallback": "torch.Tensor.copy_ / .zero_()",
        "note": "Tile size is autotuned per GPU at first JIT compile.",
    }
