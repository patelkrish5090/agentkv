"""
agentkv/hf_cache.py — HuggingFace transformers Cache backed by AgentKV pool.

This implements Phase 3a:
A `transformers.DynamicCache` subclass that stores KV data directly in an
AgentKVPool's slab blocks. This allows actual zero-copy GPU forks instead of
deep-copying the Python-level caches.
"""

from typing import Any, Dict, Optional, Tuple

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    # Fallback if transformers is not installed or too old.
    # The integration won't work, but it prevents import errors when just
    # testing the core agentkv package.
    class DynamicCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

from agentkv.pool import AgentKVPool
from agentkv.core.radix_tree import NodeHandle


class AgentKVCache(DynamicCache):
    """
    A HuggingFace Cache implementation that uses AgentKV blocks for storage.

    Instead of maintaining `key_cache` and `value_cache` as lists of 
    continuous tensors, this cache translates `update()` calls into writes
    to the shared AgentKVPool block tensors, performing CoW when necessary.
    """

    def __init__(self, pool: AgentKVPool, handle: NodeHandle) -> None:
        super().__init__()
        self.pool = pool
        self.handle = handle
        # Explicitly track how many tokens have been processed.
        # This matches handle.tokens exactly.
        self._seen_tokens = len(self.handle.tokens)
        self._next_seen_tokens = self._seen_tokens

        # We do NOT use these lists as permanent storage.
        # They remain empty.
        self.key_cache = []
        self.value_cache = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the sequence length cached so far."""
        return self._seen_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new KV states for the current layer.
        
        Args:
            key_states: [1, num_heads, num_new_tokens, head_dim]
            value_states: [1, num_heads, num_new_tokens, head_dim]
            layer_idx: index of the current layer
            
        Returns:
            Tuple of [1, num_heads, total_seq_len, head_dim] tensors 
            assembled from the AgentKV blocks.
        """
        # Batch size must be 1 for AgentKV
        if key_states.shape[0] != 1:
            raise ValueError("AgentKVCache requires batch_size=1")

        num_new_tokens = key_states.shape[2]

        # ── 1. Block Allocation (Layer 0 only) ──
        if layer_idx == 0:
            self._next_seen_tokens = self._seen_tokens + num_new_tokens
            
            # Check if we need more blocks
            required_blocks = (self._next_seen_tokens + self.pool.block_size - 1) // self.pool.block_size
            current_blocks = len(self.pool.get_block_ids(self.handle))
            
            while current_blocks < required_blocks:
                self.pool.allocate_block(self.handle)
                current_blocks += 1

            # We must also pad handle.tokens so its length matches _next_seen_tokens.
            # We append dummy token IDs (e.g. 0). They don't affect KV storage,
            # and won't be matched in the radix tree until explicitly committed 
            # with the correct tokens via pool.append_tokens().
            if len(self.handle.tokens) < self._next_seen_tokens:
                missing = self._next_seen_tokens - len(self.handle.tokens)
                self.pool.append_tokens(self.handle, [0] * missing)

        # ── 2. Write new K, V to blocks ──
        start_tok = self._seen_tokens
        
        # We loop over new tokens to place them in the correct blocks.
        # This loop is executed in Python, but num_new_tokens is usually 1 during decode.
        for i in range(num_new_tokens):
            tok_idx = start_tok + i
            block_idx = tok_idx // self.pool.block_size
            tok_in_block = tok_idx % self.pool.block_size
            
            # Identify residual block index
            residual_idx = block_idx - (self.handle.shared_match_len // self.pool.block_size)
            
            if residual_idx < 0:
                raise RuntimeError("Cannot overwrite shared prefix blocks!")

            # Check if the block is shared, and trigger CoW if needed.
            # We only need to check this on layer_idx == 0, and only once per block.
            if layer_idx == 0 and (i == 0 or tok_in_block == 0):
                self.pool.ensure_mutable_block(self.handle, residual_idx)
                
            block_id = self.pool.get_block_ids(self.handle)[block_idx]
            
            # Write KV slices
            # Slab data: [total_blocks, num_layers, 2, num_heads, block_size, head_dim]
            self.pool._allocator._kv_data[block_id, layer_idx, 0, :, tok_in_block, :] = key_states[0, :, i, :]
            self.pool._allocator._kv_data[block_id, layer_idx, 1, :, tok_in_block, :] = value_states[0, :, i, :]

        # ── 3. Assemble full sequence contiguous tensors ──
        block_ids = self.pool.get_block_ids(self.handle)
        
        # We extract all required blocks in one vectorized operation.
        # This results in a copy: [num_blocks, 2, num_heads, block_size, head_dim]
        # Using a list of IDs for advanced indexing.
        data = self.pool._allocator._kv_data[block_ids, layer_idx]
        
        num_heads = data.shape[2]
        head_dim = data.shape[4]
        
        # Separate K and V: [num_blocks, num_heads, block_size, head_dim]
        k_data = data[:, 0]
        v_data = data[:, 1]
        
        # Transpose to [num_heads, num_blocks, block_size, head_dim] 
        # and reshape to [1, num_heads, num_blocks * block_size, head_dim]
        # This automatically concatenates the block_size chunks sequentially.
        out_k = k_data.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
        out_v = v_data.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)
        
        # Truncate to exact seen tokens length
        total_len = self._next_seen_tokens
        out_k = out_k[:, :, :total_len, :]
        out_v = out_v[:, :, :total_len, :]

        # Update _seen_tokens on the last layer
        # Transformers iterates layers sequentially from 0 to num_layers-1.
        if layer_idx == self.pool.config.num_layers - 1:
            self._seen_tokens = self._next_seen_tokens

        return out_k, out_v

    def fork(self) -> 'AgentKVCache':
        """
        Create a new AgentKVCache for a child agent using zero-copy GPU CoW.
        """
        child_handle = self.pool.fork(self.handle)
        child_cache = AgentKVCache(self.pool, child_handle)
        # child_cache automatically picks up _seen_tokens = len(child_handle.tokens)
        return child_cache

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Export the cache to the legacy tuple-of-tuples format for models that 
        don't natively support the Cache protocol yet.
        """
        legacy_cache = []
        block_ids = self.pool.get_block_ids(self.handle)
        
        if not block_ids:
            # Handle empty cache case
            for _ in range(self.pool.config.num_layers):
                legacy_cache.append((
                    torch.empty(1, self.pool.config.num_kv_heads, 0, self.pool.config.head_dim, 
                                device=self.pool.config.device, dtype=self.pool._allocator._kv_data.dtype),
                    torch.empty(1, self.pool.config.num_kv_heads, 0, self.pool.config.head_dim, 
                                device=self.pool.config.device, dtype=self.pool._allocator._kv_data.dtype)
                ))
            return tuple(legacy_cache)

        for layer_idx in range(self.pool.config.num_layers):
            data = self.pool._allocator._kv_data[block_ids, layer_idx]
            num_heads = data.shape[2]
            head_dim = data.shape[4]
            
            k_data = data[:, 0]
            v_data = data[:, 1]
            
            out_k = k_data.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)[:, :, :self._seen_tokens, :]
            out_v = v_data.permute(1, 0, 2, 3).reshape(1, num_heads, -1, head_dim)[:, :, :self._seen_tokens, :]
            
            legacy_cache.append((out_k, out_v))
            
        return tuple(legacy_cache)
