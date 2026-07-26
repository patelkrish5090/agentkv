"""Stub of sglang.srt.managers.schedule_batch.Req — trimmed to the fields
integrations/sglang/radix_cache.py reads/writes."""

from typing import Any, List, Optional


class Req:
    def __init__(self, rid: str, origin_input_ids: List[int]) -> None:
        self.rid = rid
        self.origin_input_ids = list(origin_input_ids)
        self.output_ids: List[int] = []
        self.req_pool_idx: Optional[int] = None
        self.prefix_indices: Any = None
        self.last_node: Any = None
        self.num_matched_prefix_tokens: int = 0

    def fill_ids(self) -> List[int]:
        return self.origin_input_ids + self.output_ids
