import torch
from torch.utils.data import DataLoader
import json
from functools import lru_cache
from args import ArgsProto, parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from heads import get_classification_head
from mergeged_model import MergedModelBuilder
from modeling import ImageClassifier
from typing import TypedDict, Any
from tqdm import tqdm


class TaskAccuracyStat(TypedDict):
    dataset: str
    validation: str
    test: str


class AccuracyStats:
    def __init__(self, args: ArgsProto, target: str, checkpoints_dir: str):
        self._target = target
        self._model_builder = MergedModelBuilder(checkpoints_dir)
        self._program_args = args

    @lru_cache()
    def generate(self, path: str, encoding: str = "utf-8") -> list[TaskAccuracyStat]:
        stats = []
        for index, dataset in enumerate(self._model_builder.checkpoints.keys()):
            head = get_classification_head(self._program_args, dataset + "Val")
            model = self._model_builder.build(head, self._target, dataset=dataset)
            get_loader_dataset = lambda ds: get_dataset(
                ds,
                preprocess=model.train_preprocess,
                location=self._program_args.data_location,
            )

            val_loader = get_dataloader(
                get_loader_dataset(f"{dataset}Val"), is_train=False
            )
            test_loader = get_dataloader(get_loader_dataset(dataset), is_train=False)

            get_loader_stats = lambda suffix: (
                dataset + suffix,
                index + 1,
                len(self._model_builder.checkpoints),
            )
            task_accuracy_stat = {
                "dataset": dataset,
                "validation": self._evaluate_model(
                    model, val_loader, get_loader_stats("(validation)")
                ),
                "test": self._evaluate_model(
                    model, test_loader, get_loader_stats("(test)")
                ),
            }
            stats.append(task_accuracy_stat)
        with open(path, "w", encoding=encoding) as f:
            json.dump(stats, f, indent="\t")
        return stats

    def _evaluate_model(
        model: ImageClassifier,
        dataloader: DataLoader | Any,
        loader_arg: tuple[str, int, int] = ("Undefined", 0, 0),
    ) -> float:
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(
                dataloader,
                desc=f"{loader_arg[0]}({loader_arg[1]}/{loader_arg[2]})",
            ):
                images, labels = batch["images"], batch["labels"]
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return correct / total


if __name__ == "__main__":
    args = parse_arguments()
    stats = AccuracyStats(args, args.pretrained_model, args.save)
    stats = stats.generate()

    print("Generated report for following datasets:\n")
    print(f"\t{str([stat['dataset'] for stat in stats])}\n")
