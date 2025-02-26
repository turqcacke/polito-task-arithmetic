from functools import reduce
import sys
import inspect
import torch
import copy
import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Sized
from torch.utils.data import Sampler
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from datasets.cars import Cars
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100
from datasets.dtd import DTD
from datasets.eurosat import EuroSAT, EuroSATVal
from datasets.gtsrb import GTSRB
from datasets.imagenet import ImageNet
from datasets.mnist import MNIST
from datasets.resisc45 import RESISC45
from datasets.stl10 import STL10
from datasets.svhn import SVHN
from datasets.sun397 import SUN397

registry = {
    name: obj
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


class BalancedSampler(Sampler[int]):
    data_source: Sized

    def __init__(
        self,
        data_source: Sized = None,
        index_counts: Dict[Any, int] = list(),
        seed: int = None,
    ):
        if len(data_source) < len(index_counts) or len(data_source) < max(index_counts):
            raise ValueError("Indices and data source is inconsistent.")

        if not seed:
            seed = seed or int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            self._np_generator = np.random.default_rng(seed=seed)

        self.data_source = data_source
        self.index_count = index_counts
        self._min_indices_per_class = reduce(
            lambda acc, val: min(len(val), acc), index_counts.values(), float("inf")
        )
        self._len_indices = self._min_indices_per_class * len(index_counts)

    def __iter__(self) -> Iterator[int]:
        balanced_indices = []
        for indices in self.index_count.values():
            balanced_indices.extend(
                self._np_generator.choice(indices, self._min_indices_per_class)
            )
        self._np_generator.shuffle(balanced_indices)
        return iter(balanced_indices)

    def __len__(self) -> int:
        return self._len_indices


def split_train_into_train_val(
    dataset: GenericDataset,
    new_dataset_class_name: str,
    batch_size: int,
    num_workers: int,
    val_fraction: float,
    max_val_samples: int = None,
    seed: int = 0,
) -> GenericDataset:
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(
        dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed)
    )
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset: Optional[GenericDataset] = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def balance_dataset(
    dataset: GenericDataset,
    new_dataset_class_name: str,
    batch_size: int,
    num_workers: int,
    seed: int = 0,
) -> GenericDataset:
    assert dataset.train_dataset and dataset.train_loader
    dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset.train_dataset,
        dataset.train_loader.batch_size,
        shuffle=False,
        num_workers=dataset.train_loader.num_workers,
    )
    data_count: Dict[torch.TensorBase, List] = {}

    for idx, batch in tqdm(
        enumerate(dataloader),
        total=(len(dataset.train_dataset) // dataloader.batch_size)
        + (1 if len(dataset.train_dataset) % batch_size > 0 else 0),
        desc=f"Balancing[{new_dataset_class_name.replace('Balanced', '')}]",
    ):
        _, labels = batch
        offset = idx * dataloader.batch_size
        for label_idx, label in enumerate(labels):
            data_count[label.item()] = data_count.get(label.item(), list())
            data_count[label.item()].append(offset + label_idx)

    new_dataset_calss = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset: GenericDataset = new_dataset_calss()

    new_dataset.train_dataset = dataset.train_dataset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=BalancedSampler(new_dataset.train_dataset, data_count, seed),
    )
    new_dataset.test_dataset = dataset.test_dataset
    new_dataset.test_loader = dataset.test_loader

    new_dataset.classnames = copy.copy(dataset.classnames)
    return new_dataset


def get_dataset(
    dataset_name: str,
    preprocess: Any,
    location: str,
    batch_size: int = 128,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    max_val_samples: int = 5000,
    balance: bool = False,
) -> GenericDataset:
    if dataset_name.endswith("Val"):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(
                base_dataset_name,
                preprocess,
                location,
                batch_size,
                num_workers,
                balance=False,
            )
            train_val_dataset = split_train_into_train_val(
                base_dataset,
                dataset_name,
                batch_size,
                num_workers,
                val_fraction,
                max_val_samples,
            )
            dataset = (
                balance_dataset(
                    train_val_dataset,
                    dataset_name + "Balanced",
                    batch_size,
                    num_workers,
                )
                if balance
                else train_val_dataset
            )
            return dataset
    else:
        assert (
            dataset_name in registry
        ), f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
        dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, num_workers=num_workers
    )
    if balance:
        dataset = balance_dataset(
            dataset, dataset_name + "Balanced", batch_size, num_workers
        )
    return dataset
