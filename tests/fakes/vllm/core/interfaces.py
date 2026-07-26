"""Stub of vllm.core.interfaces — signatures match vllm-src/vllm/core/interfaces.py."""

import enum
from abc import ABC, abstractmethod
from typing import List
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.sequence import Sequence, SequenceGroup


class AllocStatus(enum.Enum):
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager(ABC):

    @staticmethod
    def get_block_space_manager_class(version: str):
        raise NotImplementedError("fake vllm stub: not needed for these tests")

    @abstractmethod
    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus: ...

    @abstractmethod
    def allocate(self, seq_group: SequenceGroup) -> None: ...

    @abstractmethod
    def can_append_slots(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> bool: ...

    @abstractmethod
    def append_slots(self, seq: Sequence, num_lookahead_slots: int) -> List[Tuple[int, int]]: ...

    @abstractmethod
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None: ...

    @abstractmethod
    def can_swap_in(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> AllocStatus: ...

    @abstractmethod
    def swap_in(self, seq_group: SequenceGroup, num_lookahead_slots: int) -> List[Tuple[int, int]]: ...

    @abstractmethod
    def can_swap_out(self, seq_group: SequenceGroup) -> bool: ...

    @abstractmethod
    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]: ...

    @abstractmethod
    def free(self, seq: Sequence) -> None: ...

    @abstractmethod
    def get_block_table(self, seq: Sequence) -> List[int]: ...

    @abstractmethod
    def get_num_free_gpu_blocks(self) -> int: ...

    @abstractmethod
    def get_num_free_cpu_blocks(self) -> int: ...

    @abstractmethod
    def access_all_blocks_in_seq(self, seq: Sequence, access_time: float) -> None: ...

    @abstractmethod
    def get_common_computed_block_ids(self, seqs: List[Sequence]) -> GenericSequence[int]: ...

    @abstractmethod
    def mark_blocks_as_computed(self, seq_group: SequenceGroup): ...
