"""
Packing Samplers for Efficient Batching

This module provides samplers that solve the bin packing problem for efficient batch construction.
These samplers aim to minimize computational waste by grouping variable-length samples into
fixed-size batches while preserving sampling randomness.
"""

import random
from typing import Iterator, List
from typing_extensions import override

from torch.utils.data import Sampler
from cogkit.finetune.utils import get_world_size, get_global_rank
import torch.distributed as dist


class NaivePackingSampler(Sampler):
    def __init__(self, length_list: list[int], packed_length: int, shuffle: bool = True):
        # expect length_list is a 1d int tensor
        self.length_list = length_list
        self.packed_length = packed_length
        self.num_samples = len(length_list)
        self.shuffle = shuffle

        self.idx_buckets: List[List[int]] = []
        self.flag: List[bool] = [False] * len(length_list)
        self.collected_samples = 0
        self._init_iterator()

    def _init_iterator(self):
        assert all(not flag for flag in self.flag), "All flags should be False"

        self.idx_buckets = []
        self.flag = [True] * len(self.length_list)
        self.collected_samples = 0

        shuffled_idx = list(range(len(self.length_list)))
        random.shuffle(shuffled_idx)

        while self.collected_samples < self.num_samples:
            current_length = 0
            idx_bucket = []
            for idx in shuffled_idx:
                length = self.length_list[idx]
                assert length <= self.packed_length
                assert current_length <= self.packed_length

                if self.flag[idx] is False:
                    continue

                incoming_length = current_length + length
                if incoming_length <= self.packed_length:
                    current_length = incoming_length
                    idx_bucket.append(idx)
                    self.flag[idx] = False

            self.collected_samples += len(idx_bucket)
            self.idx_buckets.append(idx_bucket)

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.idx_buckets

        if self.shuffle:
            # Shuffle the idx_buckets before the next iteration
            random.shuffle(self.idx_buckets)

            # Reset for next iteration - more aggressive approach
            # Re-initialize the buckets to introduce randomness within packed sequences
            # self.__init_iterator()

    def __len__(self):
        return len(self.idx_buckets)


class DistPackingSampler(NaivePackingSampler):
    @override
    def __init__(
        self,
        length_list: list[int],
        packed_length: int,
        shuffle: bool = True,
        world_size: int | None = None,
        global_rank: int | None = None,
    ):
        super().__init__(length_list, packed_length, shuffle)
        if not dist.is_initialized():
            raise ValueError("DistPackingSampler requires distributed training")

        self.world_size = world_size or get_world_size()
        self.global_rank = global_rank or get_global_rank()

    @override
    def __iter__(self) -> Iterator[List[int]]:
        size = len(self.idx_buckets) // self.world_size
        offset = self.global_rank * size
        yield from self.idx_buckets[offset : offset + size]

        if self.shuffle:
            random.shuffle(self.idx_buckets)

    @override
    def __len__(self):
        return len(self.idx_buckets) // self.world_size
