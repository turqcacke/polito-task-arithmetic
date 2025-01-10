import consts
import torch
import json
import os
from torch.utils.data import DataLoader
from args import ArgsProto, parse_arguments
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize
from datasets.common import get_dataloader
from heads import get_classification_head
from mergeged_model import MergedModelBuilder
from modeling import ImageClassifier, ImageEncoder
from typing import Optional, TypedDict, Any, Literal
from tqdm import tqdm
from pathlib import Path


class TaskAccuracyStat(TypedDict):
    """Wrapper for accuracy stats, can be used as `type`"""

    dataset: str
    validation: str
    test: str


# TODO: Remove if won't be used if future
class TaskAccuracyData(TypedDict):
    finetuned: Optional[TaskAccuracyStat]
    pretrained: Optional[TaskAccuracyStat]
    merged: Optional[TaskAccuracyStat]


class AccuracyStats:
    """Generates accuracy report and saves it as `json` file"""

    def __init__(
        self,
        args: ArgsProto,
        target: str,
        checkpoints_dir: str,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self._target = target
        self._model_builder = MergedModelBuilder(checkpoints_dir)
        self._program_args = args
        self._device = device

    def _get_model(
        self, dataset: str, model_type: Literal["pretrained", "finetuned", "merged"]
    ):
        head = get_classification_head(self._program_args, dataset + "Val")
        match model_type:
            case "merged":
                model = self._model_builder.build(head, self._target)
            case "finetuned":
                model = self._model_builder.build(head, self._target, dataset=dataset)
            case _:
                model = ImageClassifier(
                    ImageEncoder.load(self._program_args, self._target), head
                )
        model.to(self._device)
        return model

    def generate(
        self,
        path: str,
        model_type: consts.SINGLE_TASK_MODEL_TYPES,
        encoding: str = "utf-8",
    ) -> list[TaskAccuracyStat]:
        """Generation method, used to generate `json`
        and start evaluation

        :param path: Path where to save `json` report
        :type path: str
        :param model_type: Specifies which model checkpoint to use
        :type model_type: consts.SINGLE_TASK_MODEL_TYPES
        :param encoding: `json` file encoding, defaults to "utf-8"
        :type encoding: str, optional
        :return: List of task based reports
        :rtype: list[TaskAccuracyStat]
        """
        stats = []
        for index, dataset in enumerate(self._model_builder.checkpoints.keys()):
            model = self._get_model(dataset, model_type)
            get_loader_dataset = lambda ds: get_dataset(
                ds,
                preprocess=model.train_preprocess,
                location=self._program_args.data_location,
            )

            val_loader = get_dataloader(
                get_loader_dataset(f"{dataset}Val"),
                args=self._program_args,
                is_train=False,
            )
            test_loader = get_dataloader(
                get_loader_dataset(dataset),
                args=self._program_args,
                is_train=False,
            )

            get_loader_stats = lambda suffix: (
                dataset + suffix,
                index + 1,
                len(self._model_builder.checkpoints),
            )
            task_accuracy_stat = TaskAccuracyStat(
                dataset=dataset,
                validation=self._evaluate_model(
                    model,
                    val_loader,
                    get_loader_stats(f"(validation)[{model_type}]"),
                ),
                test=self._evaluate_model(
                    model, test_loader, get_loader_stats(f"(test)[{model_type}]")
                ),
            )
            stats.append(task_accuracy_stat)
        with open(path, "w", encoding=encoding) as f:
            json.dump(stats, f, indent="\t")
        return stats

    def _evaluate_model(
        self,
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
                data = maybe_dictionarize(batch)

                images = data["images"].to(self._device)
                labels = data["labels"].to(self._device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return correct / total


if __name__ == "__main__":
    args = parse_arguments()

    checkpoints_dir = Path(args.save)
    save_dir = consts.BASE_DIR / consts.EVAL_FOLDER
    model_type = args.st_model

    os.makedirs(str(save_dir), exist_ok=True)

    save_dir = save_dir / consts.SINGLE_TASK_SAVE_FILE.format(suffix=f"_{model_type}")
    stats = AccuracyStats(
        args, str(checkpoints_dir / consts.PRETRAINED_MODEL_NAME), str(checkpoints_dir)
    )

    stats = stats.generate(str(save_dir), args.st_model)

    print("\nGenerated report for following datasets:")
    print(f"\t{str([stat['dataset'] for stat in stats])}")
    print(f"\t Report saved to `{str(save_dir)}`")
