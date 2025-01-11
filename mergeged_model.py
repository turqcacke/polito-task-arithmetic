import consts
import os
from dataclasses import dataclass
from modeling import ImageClassifier, ClassificationHead
from task_vectors import NonLinearTaskVector
from typing import Callable, Dict, Optional, Sequence
from tqdm import tqdm
from pathlib import Path


@dataclass
class CheckpointPath:
    head: str | None = None
    encoder: str | None = None


class MergedModelBuilder:
    def __init__(self, checkpoints_dir: str):
        self._checkpoints_dir = checkpoints_dir
        self._vectors_cache: Dict[str, NonLinearTaskVector] = {}
        self.checkpoints = self._get_checkpoint_path()

    def _load_tasks(
        self, pretrained: str, filter_: Callable[[str], bool]
    ) -> Dict[str, NonLinearTaskVector]:
        tasks = {}
        for dataset, checkpoint in tqdm(
            self.checkpoints.items(), desc="Load task vector"
        ):
            if not filter_(dataset):
                continue
            encoder = NonLinearTaskVector(pretrained, checkpoint.encoder)
            tasks[dataset] = encoder
        return tasks

    def _get_checkpoint_path(
        self, exclude: Sequence[str] = consts.EXCLUDED_CHECKPOINTS
    ) -> Dict[str, CheckpointPath]:
        if not os.path.exists(self._checkpoints_dir):
            raise FileNotFoundError()
        base_dir = Path(self._checkpoints_dir)
        head_prefix = "head_"
        head_suffix = "Val.pt"
        encoder_suffix = "_finetuned.pt"
        extract_dataset_name: Callable[[str], str] = lambda f: (
            f.removeprefix(head_prefix).removesuffix(head_suffix)
            if f.startswith(head_prefix)
            else f.removesuffix(encoder_suffix)
        )
        checkpoints: Dict[str, CheckpointPath] = {}
        for file in os.listdir(str(base_dir)):
            if file in exclude:
                continue
            file_path = str(base_dir / file)
            dataset = extract_dataset_name(file)
            checkpoint_path = checkpoints.get(dataset, CheckpointPath())
            if file.startswith(head_prefix):
                checkpoint_path.head = file_path
                continue
            checkpoint_path.encoder = file_path
            checkpoints[dataset] = checkpoint_path
        return checkpoints

    def build(
        self,
        head: ClassificationHead,
        pretrained_model: str,
        alpha: float = consts.DEFAULT_ALPHA,
        dataset: Optional[str] = None,
    ) -> ImageClassifier:
        vector_tasks = self._vectors_cache.get(dataset, None) or self._load_tasks(
            pretrained_model,
            lambda name: name == dataset if dataset else lambda _: True,
        )
        add_vector = None

        if not vector_tasks:
            raise AssertionError("No task defined with provided filter.")

        if isinstance(vector_tasks, NonLinearTaskVector):
            vector_tasks = {None: vector_tasks}

        for task in vector_tasks.values():
            if not add_vector:
                add_vector = task
                continue
            add_vector += task

        if dataset not in self._vectors_cache:
            self._vectors_cache[dataset] = add_vector

        merged_ecoder = add_vector.apply_to(pretrained_model, scaling_coef=alpha)
        return ImageClassifier(merged_ecoder, head)
