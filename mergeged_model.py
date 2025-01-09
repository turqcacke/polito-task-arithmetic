import os
import torch.nn.modules
from args import ArgsProto
from modeling import ImageEncoder, ImageClassifier, ClassificationHead
from task_vectors import NonLinearTaskVector
from typing import NamedTuple, Callable, Dict, Optional, Any
from pathlib import Path


class CheckpointPath(NamedTuple):
    head: str | None
    encoder: str | None


class MergedModelBuilder:
    MODEL_NAME = "MERGED_model.pt"

    def __init__(self, checkpoints_dir: str):
        self._tasks: Dict[str, NonLinearTaskVector] = {}
        self._checkpoints_dir = checkpoints_dir
        self.checkpoints = self._get_checkpoint_path()

    def _load_pretrained_models(self, pretrained: str, filter_: Callable[[str], bool]):
        for dataset, checkpoint in self.checkpoints.items():
            if not filter_(dataset):
                continue
            encoder = NonLinearTaskVector(pretrained, checkpoint.encoder)
            self._tasks[dataset] = encoder

    def _get_checkpoint_path(self) -> Dict[str, CheckpointPath]:
        if not os.path.exists(self._checkpoints_dir):
            raise FileNotFoundError()
        base_dir = Path(self._checkpoints_dir)
        head_prefix = "head"
        head_suffix = "Val.pt"
        encoder_suffix = "finetuned.pt"
        extract_dataset_name: Callable[[str], str] = lambda f: (
            f.removeprefix(head_prefix).removesuffix(head_suffix)
            if f.startswith(head_prefix)
            else f.removesuffix(encoder_suffix)
        )
        checkpoints: Dict[str, CheckpointPath] = {}
        for file in os.listdir(str(base_dir)):
            if file == self.MODEL_NAME:
                continue
            file_path = str(base_dir / file)
            dataset = extract_dataset_name(file)
            checkpoint_path = checkpoints.get(dataset, CheckpointPath(None, None))
            if file.startswith(head_prefix):
                checkpoint_path.head = file_path
                continue
            checkpoint_path.encoder = file_path
        return checkpoints

    def build(
        self,
        head: ClassificationHead,
        pretrained_model: str,
        alpha: float = 1.0,
        dataset: Optional[str] = None,
    ) -> ImageClassifier:
        self._load_pretrained_models(
            pretrained_model,
            lambda name: name == dataset if dataset else lambda _: True,
        )
        add_vector = None

        for task in self._tasks.values():
            if not add_vector:
                add_vector = task
                continue
            add_vector += task

        merged_ecoder = add_vector.apply_to(pretrained_model, scaling_coef=alpha)
        return ImageClassifier(merged_ecoder, head)
