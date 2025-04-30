import random

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from cogkit.finetune.samplers import NaivePackingSampler


# ==============================================================================
# initialization and iterator test
# ==============================================================================


@pytest.fixture
def basic_length_list():
    return torch.tensor([3, 5, 2, 4, 1, 6, 3, 2])


def test_initialization(basic_length_list):
    """Test the initialization of the sampler"""
    packed_length = 16
    sampler = NaivePackingSampler(basic_length_list, packed_length)

    assert sampler.num_samples == 8
    assert sampler.packed_length == 16
    assert len(sampler.idx_buckets) > 0


@pytest.mark.parametrize("packed_length,num_samples", [(512, 100), (1024, 500), (2048, 1000)])
def test_packing_constraint(packed_length, num_samples):
    """Test packing constraint with different combinations of packed_length and num_samples"""

    data = [random.randint(1, packed_length // 4) for _ in range(num_samples)]
    random_lengths = torch.tensor(data)
    sampler = NaivePackingSampler(random_lengths, packed_length)

    collected_samples = set()
    for idx_bucket in sampler.idx_buckets:
        bucket_lengths = torch.tensor([random_lengths[i].item() for i in idx_bucket])
        assert bucket_lengths.sum().item() <= packed_length

        # Ensure each collated idx is unique
        for idx in idx_bucket:
            assert idx not in collected_samples
            collected_samples.add(idx)

    assert len(collected_samples) == num_samples


# ==============================================================================
# dataloader test
# ==============================================================================


class CustomTestDataset(Dataset):
    def __init__(self, lengths):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.lengths[idx]


@pytest.fixture(params=[(11, 333), (50, 256), (77, 33), (100, 512), (200, 4), (500, 1024)])
def packing_test_config(request):
    num_samples, packed_length = request.param

    data = [random.randint(1, packed_length // random.randint(1, 4)) for _ in range(num_samples)]
    random_lengths = torch.tensor(data)

    dataset = CustomTestDataset(random_lengths)

    return {
        "num_samples": num_samples,
        "packed_length": packed_length,
        "random_lengths": random_lengths,
        "dataset": dataset,
    }


def test_dataloader_packing_constraint(packing_test_config):
    """Test if the sampler maintains packing constraints when used with DataLoader"""
    random_lengths = packing_test_config["random_lengths"]
    packed_length = packing_test_config["packed_length"]
    dataset = packing_test_config["dataset"]

    sampler = NaivePackingSampler(random_lengths, packed_length=packed_length, shuffle=True)
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    values = set(v.item() for v in random_lengths)

    for batch in dataloader:
        # Assert that each element in the batch exists in the original values
        for v in batch:
            assert v.item() in values, (
                f"Found value {v} in batch that doesn't exist in original data"
            )

        assert torch.sum(batch) <= packed_length, (
            f"Batch total length ({torch.sum(batch)}) exceeds constraint ({packed_length})"
        )


def test_shuffle_consistency(packing_test_config):
    """Test if shuffling=False maintains consistent iteration results"""
    random_lengths = packing_test_config["random_lengths"]
    packed_length = packing_test_config["packed_length"]
    dataset = packing_test_config["dataset"]

    # Test shuffle=False
    sampler_no_shuffle = NaivePackingSampler(
        random_lengths, packed_length=packed_length, shuffle=False
    )
    dataloader_no_shuffle = DataLoader(dataset, batch_sampler=sampler_no_shuffle)

    # First iteration
    first_iteration = []
    for batch in dataloader_no_shuffle:
        first_iteration.append([x.item() for x in batch])

    # Second iteration
    second_iteration = []
    for batch in dataloader_no_shuffle:
        second_iteration.append([x.item() for x in batch])

    # Compare two iterations
    assert len(first_iteration) == len(second_iteration), (
        "The number of batches in two iterations should be the same"
    )
    for i, (batch1, batch2) in enumerate(zip(first_iteration, second_iteration)):
        assert batch1 == batch2, f"When shuffle=False, the {i}th batch should be the same"


def test_shuffle_variation(packing_test_config):
    """Test if shuffling=True produces different iteration results"""
    random_lengths = packing_test_config["random_lengths"]
    packed_length = packing_test_config["packed_length"]
    dataset = packing_test_config["dataset"]

    # Test shuffle=True
    sampler_shuffle = NaivePackingSampler(random_lengths, packed_length=packed_length, shuffle=True)
    dataloader_shuffle = DataLoader(dataset, batch_sampler=sampler_shuffle)

    # First iteration
    first_iteration = []
    for batch in dataloader_shuffle:
        first_iteration.append([x.item() for x in batch])

    # Second iteration
    second_iteration = []
    for batch in dataloader_shuffle:
        second_iteration.append([x.item() for x in batch])

    # Check if at least one batch is different
    any_different = False
    for batch1, batch2 in zip(first_iteration, second_iteration):
        if batch1 != batch2:
            any_different = True
            break

    assert any_different, "When shuffle=True, at least one batch should be different"
