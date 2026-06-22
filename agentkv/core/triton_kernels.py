"""
agentkv/core/triton_kernels.py — Triton device kernels for KV block operations.

Target: RTX 5000 Ada (sm_89, Ada Lovelace architecture).

Kernels provided
----------------
  copy_block_kernel   : Copy one KV block to another in GPU VRAM.
                        Used during Copy-on-Write when a shared block is
                        about to be written by a single agent.

  zero_block_kernel   : Zero-fill a KV block.
                        Used after allocation to ensure no data leakage
                        between agents.

Design notes
------------
- We use Triton's tl.load / tl.store rather than raw CUDA C++ for portability
  and to avoid requiring a CUDA C++ toolchain.
- Block granularity: each kernel instance processes one KV block.
  A KV block has shape [num_layers, 2, num_kv_heads, block_size, head_dim],
  which we flatten to a 1D vector for the kernel.
- BLOCK_SIZE_K (Triton tile size, not the KV block_size) is tuned for sm_89:
  128 elements per warp × 4 warps = 512 elements per tile.  This fills
  the L1 cache line well on Ada's 128-byte cache lines.
- On CPU (no Triton), we fall back to torch.Tensor.copy_ / .zero_(), which
  is correct but slower.

GPU availability detection
--------------------------
We detect Triton availability at import time.  If Triton is not available
(e.g. native Windows, CPU-only env), all functions fall back to PyTorch.
This makes the full test suite runnable without a GPU.
"""

from __future__ import annotations

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


# ── Triton kernel definitions (only compiled when Triton is available) ─────────

if _TRITON_AVAILABLE:

    @triton.jit
    def _copy_block_kernel(
        src_ptr,
        dst_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Copy n_elements floats from src_ptr to dst_ptr.

        Launch config: grid = (cdiv(n_elements, BLOCK_SIZE),), 1 program.
        Each program handles BLOCK_SIZE elements.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        data = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(dst_ptr + offsets, data, mask=mask)

    @triton.jit
    def _zero_block_kernel(
        dst_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Zero-fill n_elements floats at dst_ptr."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(dst_ptr + offsets, tl.zeros([BLOCK_SIZE], dtype=tl.float16), mask=mask)

else:
    _copy_block_kernel = None  # type: ignore[assignment]
    _zero_block_kernel = None  # type: ignore[assignment]


# ── sm_89 (Ada Lovelace) tuning constants ─────────────────────────────────────
# Ada has 128 B cache lines and 32 B per half-precision element (fp16).
# 512 fp16 elements = 1 KB = 8 cache lines — a good L1 working set.
_TRITON_BLOCK_SIZE = 512  # elements per Triton program tile


# ── Public dispatch functions ─────────────────────────────────────────────────

def copy_block(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy a KV block tensor from src to dst.

    Parameters
    ----------
    src : torch.Tensor
        Source block.  Shape: [num_layers, 2, num_kv_heads, block_size, head_dim].
        Must be contiguous and on CUDA.
    dst : torch.Tensor
        Destination block.  Same shape as src.  Must be contiguous and on CUDA.
        Caller guarantees exclusive ownership (ref_count == 1).
    """
    assert src.shape == dst.shape, f"Shape mismatch: {src.shape} vs {dst.shape}"

    if src.device.type == "cpu" or not _TRITON_AVAILABLE:
        # CPU path: use PyTorch (correct, slower)
        dst.copy_(src)
        return

    # GPU path: Triton kernel
    src_flat = src.view(-1)
    dst_flat = dst.view(-1)
    n = src_flat.numel()

    # We need to work in float16 or bfloat16 for the kernel.
    # Cast if needed (copy_ handles this on CPU; on GPU we do it explicitly).
    if src_flat.dtype not in (torch.float16, torch.bfloat16):
        dst.copy_(src)
        return

    grid = (triton.cdiv(n, _TRITON_BLOCK_SIZE),)
    _copy_block_kernel[grid](
        src_flat,
        dst_flat,
        n,
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
    )


def zero_block(block: torch.Tensor) -> None:
    """Zero-fill a KV block tensor.

    Parameters
    ----------
    block : torch.Tensor
        Block to zero.  Must be contiguous.
        Caller guarantees exclusive ownership.
    """
    if block.device.type == "cpu" or not _TRITON_AVAILABLE:
        block.zero_()
        return

    flat = block.view(-1)
    n = flat.numel()

    if flat.dtype not in (torch.float16, torch.bfloat16):
        block.zero_()
        return

    grid = (triton.cdiv(n, _TRITON_BLOCK_SIZE),)
    _zero_block_kernel[grid](
        flat,
        n,
        BLOCK_SIZE=_TRITON_BLOCK_SIZE,
    )


def is_triton_available() -> bool:
    """Returns True if Triton is installed and GPU kernels will be used."""
    return _TRITON_AVAILABLE


def kernel_info() -> dict:
    """Return information about the current kernel configuration."""
    return {
        "triton_available": _TRITON_AVAILABLE,
        "triton_block_size": _TRITON_BLOCK_SIZE,
        "target_arch": "sm_89 (Ada Lovelace)",
        "fallback": "torch.Tensor.copy_ / .zero_()",
    }
