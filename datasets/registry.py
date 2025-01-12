from collections import Counter
import sys
import inspect
import random
import torch
import copy
import numpy as np

from typing import Any, Optional
from torch.utils.data import Subset
from torch.utils.data.dataset import random_split

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
) -> GenericDataset:
    assert dataset.train_dataset

    new_dataset_calss = type(new_dataset_class_name, (GenericDataset), {})
    new_dataset: GenericDataset = new_dataset_calss()

    for split_type in ("train", "test"):
        loader_attr = f"{split_type}_loader"
        dataset_attr = f"{split_type}_dataset"

        labels = [label for _, label in getattr(dataset, dataset_attr)]
        class_counts = Counter(labels)
        min_class_count = min(class_counts.values())

        balanced_indices = []
        for class_label in class_counts:
            class_indices = [
                i for i, (_, label) in enumerate(dataset) if label == class_label
            ]
            sampled_indices = np.random.choice(
                class_indices, min_class_count, replace=False
            )
            balanced_indices += sampled_indices

        new_dataset.__setattr__(dataset_attr, Subset(dataset, balanced_indices))
        new_dataset.__setattr__(
            loader_attr,
            torch.utils.data.DataLoader(
                getattr(new_dataset, dataset_attr),
                batch_size=batch_size,
                num_workers=num_workers,
            ),
        )
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
                balance,
            )
            dataset = split_train_into_train_val(
                base_dataset,
                dataset_name,
                batch_size,
                num_workers,
                val_fraction,
                max_val_samples,
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
