"""
agentkv/core/config.py — Pool configuration dataclass.

PoolConfig is the single source of truth for all allocator parameters.
Passing a PoolConfig through every layer avoids proliferating keyword
arguments and makes configuration serializable.

Design note
-----------
We deliberately keep this as a plain dataclass (not a Pydantic model) to
keep the dependency footprint minimal.  Validation is done in __post_init__.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PoolConfig:
    """Configuration for an AgentKV memory pool.

    Attributes
    ----------
    total_blocks : int
        Number of KV blocks pre-allocated in the pool.  Each block holds
        ``block_size`` tokens across all layers and KV heads.
        Must be a power-of-two for efficient slab math (validated below).
    block_size : int
        Tokens per block (default 16, matching vLLM's PagedAttention default).
    num_layers : int
        Number of transformer layers.  Used to compute per-block byte size.
    num_kv_heads : int
        Number of KV attention heads per layer.
    head_dim : int
        Dimension per head.
    dtype : str
        Tensor dtype as a string: 'float16', 'bfloat16', or 'float32'.
        Used to compute per-element byte size.
    device : str
        PyTorch device string, e.g. 'cuda', 'cuda:0', 'cpu'.
        CPU device is supported for testing without a GPU.
    max_agents : int
        Maximum number of concurrent agent handles the pool supports.
        Sets the size of the per-agent metadata arrays.
    epoch_interval : int
        How often (in allocation operations) the epoch reclaimer advances
        the epoch counter to retire freed blocks.  Lower = more frequent
        reclamation checks; higher = less overhead but delayed block recycling.
    """

    total_blocks: int = 4096
    block_size: int = 16
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: str = "cuda"
    max_agents: int = 1024
    epoch_interval: int = 64

    # ── computed / internal fields ────────────────────────────────────────────
    # These are set in __post_init__ and should not be supplied by callers.
    bytes_per_element: int = field(init=False, repr=False)
    bytes_per_block: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # --- dtype validation ---
        _dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}
        if self.dtype not in _dtype_bytes:
            raise ValueError(
                f"dtype must be one of {list(_dtype_bytes)}, got '{self.dtype}'"
            )
        self.bytes_per_element = _dtype_bytes[self.dtype]

        # --- block size validation ---
        if self.block_size <= 0 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError(
                f"block_size must be a positive power of 2, got {self.block_size}"
            )

        # --- total_blocks validation ---
        if self.total_blocks <= 0:
            raise ValueError(f"total_blocks must be > 0, got {self.total_blocks}")

        # --- max_agents ---
        if self.max_agents <= 0:
            raise ValueError(f"max_agents must be > 0, got {self.max_agents}")

        # --- per-block byte size ---
        # Layout: [num_layers, 2 (K+V), num_kv_heads, block_size, head_dim]
        # Factor of 2 for key + value tensors.
        elements_per_block = (
            self.num_layers * 2 * self.num_kv_heads * self.block_size * self.head_dim
        )
        self.bytes_per_block = elements_per_block * self.bytes_per_element

    @property
    def total_bytes(self) -> int:
        """Total bytes required for the KV data pool."""
        return self.total_blocks * self.bytes_per_block

    @property
    def total_gb(self) -> float:
        """Total pool size in GiB."""
        return self.total_bytes / (1024**3)

    @classmethod
    def from_capacity_gb(
        cls,
        capacity_gb: float,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: str = "float16",
        device: str = "cuda",
        **kwargs,
    ) -> "PoolConfig":
        """Construct a PoolConfig from a target GPU memory budget in GiB.

        The number of blocks is computed so that the pool fits within
        ``capacity_gb`` GiB. The result is rounded *down* to the nearest
        power of two to keep slab alignment simple.

        Parameters
        ----------
        capacity_gb : float
            Target pool size in GiB (e.g. 40.0 for a 40 GB A100 / Ada GPU).
        """
        _dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}
        bytes_per_elem = _dtype_bytes[dtype]
        elements_per_block = num_layers * 2 * num_kv_heads * block_size * head_dim
        bytes_per_block = elements_per_block * bytes_per_elem

        total_bytes = int(capacity_gb * (1024**3))
        raw_blocks = total_bytes // bytes_per_block

        # Round down to the nearest power of two (≥ 1)
        if raw_blocks <= 0:
            raise ValueError(
                f"capacity_gb={capacity_gb} is too small for the given block geometry"
            )
        total_blocks = 2 ** math.floor(math.log2(raw_blocks))

        return cls(
            total_blocks=total_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    @classmethod
    def max_for_device(
        cls,
        fraction: float = 0.7,
        block_size: int = 16,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype: str = "float16",
        device: str = "cuda",
        **kwargs,
    ) -> "PoolConfig":
        """Construct a PoolConfig that uses ``fraction`` of available GPU VRAM.

        Queries ``torch.cuda.mem_get_info()`` to find free VRAM and sizes the
        pool to ``fraction * free_vram``.  This is the safest way to create a
        pool on a shared GPU (e.g. Google Colab T4) without hitting OOM.

        Parameters
        ----------
        fraction : float
            Fraction of *currently free* VRAM to use (default 0.7 = 70%).
            Keep < 1.0 to leave room for PyTorch's own allocator overhead,
            Triton's JIT cache, and the model weights if co-located.

        Example
        -------
        >>> cfg = PoolConfig.max_for_device(fraction=0.6)
        >>> print(cfg)  # shows how many blocks fit in 60% of free VRAM
        """
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "max_for_device() requires a CUDA GPU. "
                "Use PoolConfig(total_blocks=...) for CPU pools."
            )

        free_bytes, total_bytes = torch.cuda.mem_get_info(device if device != "cpu" else 0)
        budget_bytes = free_bytes * fraction
        budget_gb = budget_bytes / (1024 ** 3)

        return cls.from_capacity_gb(
            capacity_gb=budget_gb,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            f"PoolConfig("
            f"total_blocks={self.total_blocks}, "
            f"block_size={self.block_size}, "
            f"layers={self.num_layers}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"pool_size={self.total_gb:.2f} GiB"
            f")"
        )
