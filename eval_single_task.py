import consts
import torch
import json
import os
import utils
from torch.utils.data import DataLoader
from args import ArgsProto, parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from heads import get_classification_head
from mergeged_model import MergedModelBuilder
from modeling import ImageClassifier, ImageEncoder
from typing import Dict, List, Optional, Tuple, TypedDict, Any, Literal
from pathlib import Path


class Stat(TypedDict):
    absolute: Optional[float]
    normalized: Optional[float]


class TaskAccuracyStat(TypedDict):
    """Wrapper for accuracy stats, can be used as `type`"""

    dataset: str
    train: Stat
    test: Stat


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
        alpha = self._program_args.st_alpha or consts.DEFAULT_ALPHA
        match model_type:
            case "merged":
                model = self._model_builder.build(head, self._target, alpha)
            case "finetuned":
                model = self._model_builder.build(head, self._target, alpha, dataset)
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
        normalized_idvisors: Dict[str, float] = {},
        save=False,
        encoding: str = "utf-8",
    ) -> List[TaskAccuracyStat]:
        """Generation method, used to generate `json`
        and start evaluation

        :param path: Path where to save `json` report
        :type path: str
        :param model_type: Specifies which model checkpoint to use
        :type model_type: consts.SINGLE_TASK_MODEL_TYPES
        :param save: Whether to save to a file or not, defaults to False
        :type save: bool, optional
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
            get_stat = lambda *args, **kwargs: Stat(
                **{
                    t[0]: t[1]
                    for t in zip(
                        ("absolute", "normalized"),
                        self._evaluate_model(
                            *args,
                            **kwargs,
                        ),
                    )
                }
            )
            task_accuracy_stat = TaskAccuracyStat(
                dataset=dataset,
                train=get_stat(
                    model,
                    val_loader,
                    norm_divisor=normalized_idvisors.get(dataset),
                    loader_args=get_loader_stats(f"(train)[{model_type}]"),
                ),
                test=get_stat(
                    model,
                    test_loader,
                    norm_divisor=normalized_idvisors.get(dataset),
                    loader_args=get_loader_stats(f"(test)[{model_type}]"),
                ),
            )
            stats.append(task_accuracy_stat)

        if save:
            with open(path, "w", encoding=encoding) as f:
                json.dump(stats, f, indent="\t")

        return stats

    def _evaluate_model(self, *args, **kwargs) -> Tuple[float, float]:
        return utils.evaluate_model(*args, **kwargs)


if __name__ == "__main__":
    args = parse_arguments()

    checkpoints_dir = Path(args.save)
    save_dir = consts.BASE_DIR / consts.EVAL_FOLDER
    alpha = args.st_alpha

    model_type = args.st_model
    alpha_suffix = f"_{alpha:.2f}" if alpha else ""

    os.makedirs(str(save_dir), exist_ok=True)

    save_dir = save_dir / consts.SINGLE_TASK_SAVE_FILE.format(
        suffix=f"_{model_type}{alpha_suffix}"
    )
    stats = AccuracyStats(
        args, str(checkpoints_dir / consts.PRETRAINED_MODEL_NAME), str(checkpoints_dir)
    )

    stats = stats.generate(str(save_dir), args.st_model, save=True)

    print("\nGenerated report for following datasets:")
    print(f"\t{str([stat['dataset'] for stat in stats])}")
    print(f"\t Report saved to `{str(save_dir)}`")
