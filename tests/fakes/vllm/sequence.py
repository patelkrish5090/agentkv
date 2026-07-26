"""Stub of vllm.sequence — trimmed to the surface AgentKVBlockManager touches.

Field/method names match vllm-src/vllm/sequence.py (Sequence.seq_id,
Sequence.get_token_ids, SequenceStatus.WAITING/RUNNING,
SequenceGroup.get_seqs(status=...)) so the real integration code runs
unmodified against this stub.
"""

import enum
from typing import Dict, List, Optional


class SequenceStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    FINISHED = enum.auto()


class Sequence:
    def __init__(self, seq_id: int, token_ids: List[int]) -> None:
        self.seq_id = seq_id
        self._token_ids = list(token_ids)
        self.status = SequenceStatus.WAITING

    def get_token_ids(self) -> List[int]:
        return self._token_ids

    def append_token_id(self, token_id: int) -> None:
        self._token_ids.append(token_id)


class SequenceGroup:
    def __init__(self, seqs: List[Sequence]) -> None:
        self.seqs_dict: Dict[int, Sequence] = {s.seq_id: s for s in seqs}

    def get_seqs(self, status: Optional[SequenceStatus] = None) -> List[Sequence]:
        if status is None:
            return list(self.seqs_dict.values())
        return [s for s in self.seqs_dict.values() if s.status == status]
