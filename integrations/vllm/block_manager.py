import typing
from typing import List, Tuple, Optional, Sequence as GenericSequence

from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

from agentkv import AgentKVPool, PoolConfig
from agentkv.core.radix_tree import NodeHandle

class AgentKVBlockManager(BlockSpaceManager):
    """
    A drop-in replacement for vLLM's BlockSpaceManagerV2.
    
    Instead of using vLLM's internal prefix tree and PrefixCachingBlockAllocator,
    this uses AgentKVPool's DualRadixTree and SlabAllocator. 
    It intercepts CoW (Copy-on-Write) events and returns them so vLLM's 
    CacheEngine can perform physical tensor copies.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = True,
    ) -> None:
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_cpu_blocks = num_cpu_blocks
        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.sliding_window = sliding_window

        # Initialize AgentKVPool with "meta" device to avoid double-allocating VRAM.
        # vLLM's CacheEngine manages the physical memory block tensors.
        # AgentKV manages the logical tree and block lifetime epochs.
        cfg = PoolConfig(
            total_blocks=num_gpu_blocks,
            block_size=block_size,
            num_layers=1,  # Not needed for metadata
            num_kv_heads=1,
            head_dim=1,
            device="meta",
            dtype="float16"
        )
        self.pool = AgentKVPool(config=cfg)
        
        # We need to track CoW events for CacheEngine.
        # Patch the slab allocator to record CoWs instead of performing physical copy.
        self.pool._allocator.pending_cows = []
        original_copy_block = self.pool._allocator.copy_block
        
        def mock_copy_block(src_idx: int, dst_idx: int):
            # Record the cow
            self.pool._allocator.pending_cows.append((src_idx, dst_idx))
            # No physical copy since device="meta"
        
        self.pool._allocator.copy_block = mock_copy_block

        # seq.seq_id -> NodeHandle
        self.seq_to_handle: typing.Dict[int, NodeHandle] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        token_ids = seq.get_token_ids()
        
        # Calculate blocks required for prompt
        required_blocks = (len(token_ids) + self.block_size - 1) // self.block_size
        
        if self.sliding_window is not None:
            max_blocks = (self.sliding_window + self.block_size - 1) // self.block_size + 1
            required_blocks = min(required_blocks, max_blocks)
            
        free_blocks = self.get_num_free_gpu_blocks()
        
        if self.num_gpu_blocks - required_blocks < self.watermark_blocks:
            return AllocStatus.NEVER
        if free_blocks - required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        
        # Assume all sequences in group share the same prompt
        seq = waiting_seqs[0]
        token_ids = seq.get_token_ids()
        
        # AgentKV handles prefix matching automatically in create_root
        handle = self.pool.create_root(token_ids)
        self.seq_to_handle[seq.seq_id] = handle
        
        # Allocate blocks to fit the prompt
        # create_root might have already matched a shared prefix, so handle.tokens could be > 0
        # Wait, create_root does NOT append tokens or allocate blocks in AgentKVPool
        # unless we explicitly append. 
        # But create_root initializes handle.tokens with the matched prefix.
        
        current_blocks = len(self.pool.get_block_ids(handle))
        required_blocks = (len(token_ids) + self.block_size - 1) // self.block_size
        
        # If we need more blocks beyond the shared prefix
        while current_blocks < required_blocks:
            self.pool.allocate_block(handle)
            current_blocks += 1
            
        # Optional: we can commit the prompt to the shared tree immediately 
        # so future requests share it. We commit up to the last full block.
        safe_share_len = (len(token_ids) // self.block_size) * self.block_size
        if safe_share_len > 0:
            self.pool.commit_prefix(handle, safe_share_len)

        # Fork for speculative decoding or beam search
        for child_seq in waiting_seqs[1:]:
            child_handle = self.pool.fork(handle)
            self.seq_to_handle[child_seq.seq_id] = child_handle

    def can_append_slots(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> bool:
        # A simple worst-case check
        running_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        num_touched_blocks = len(running_seqs) * (1 + num_lookahead_slots)
        return self.get_num_free_gpu_blocks() >= num_touched_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        handle = self.seq_to_handle[seq.seq_id]
        self.pool._allocator.pending_cows.clear()
        
        # 1. Update the tree with new generated tokens
        current_len = len(handle.tokens)
        new_tokens = seq.get_token_ids()[current_len:]
        
        if new_tokens:
            self.pool.append_tokens(handle, new_tokens)
            
            # Check if we need a new block for the new tokens
            current_blocks = len(self.pool.get_block_ids(handle))
            required_blocks = (len(handle.tokens) + self.block_size - 1) // self.block_size
            while current_blocks < required_blocks:
                self.pool.allocate_block(handle)
                current_blocks += 1

            # 2. Trigger CoW on the last block if it was shared.
            # This must happen because we are appending/modifying it.
            if len(handle.tokens) > 0:
                last_block_idx = (len(handle.tokens) - 1) // self.block_size
                residual_idx = last_block_idx - (handle.shared_match_len // self.block_size)
                if residual_idx >= 0:
                    self.pool.ensure_mutable_block(handle, residual_idx)
        
        # 3. Retrieve any physical CoW operations generated
        cows = list(self.pool._allocator.pending_cows)
        self.pool._allocator.pending_cows.clear()
        return cows

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        parent_handle = self.seq_to_handle[parent_seq.seq_id]
        child_handle = self.pool.fork(parent_handle)
        self.seq_to_handle[child_seq.seq_id] = child_handle

    def free(self, seq: Sequence) -> None:
        if seq.seq_id in self.seq_to_handle:
            handle = self.seq_to_handle.pop(seq.seq_id)
            self.pool.free(handle)

    def get_block_table(self, seq: Sequence) -> List[int]:
        handle = self.seq_to_handle[seq.seq_id]
        return self.pool.get_block_ids(handle)

    def get_num_free_gpu_blocks(self) -> int:
        return self.pool._allocator.free_blocks

    def get_num_free_cpu_blocks(self) -> int:
        return self.num_cpu_blocks  # CPU block swapping not supported in AgentKV demo

    def can_swap_in(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> AllocStatus:
        return AllocStatus.NEVER

    def swap_in(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swapping not implemented in AgentKVBlockManager")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        return False

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        raise NotImplementedError("Swapping not implemented in AgentKVBlockManager")

    def access_all_blocks_in_seq(self, seq: Sequence, access_time: float) -> None:
        pass

    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> GenericSequence[int]:
        # Assume 0 common computed blocks for skip-prefill logic
        return []

    def mark_blocks_as_computed(self, seq_group: SequenceGroup):
        pass
