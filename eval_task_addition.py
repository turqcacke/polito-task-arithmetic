from pathlib import Path
import consts
import json
import os
import torch
import utils
from args import ArgsProto, parse_arguments
from datasets.registry import get_dataset
from datasets.common import get_dataloader
from eval_single_task import AccuracyStats
from heads import get_classification_head
from mergeged_model import MergedModelBuilder
from typing import Dict, List, Optional, Union
from tqdm import tqdm


class MultiTaskAccuracyStats:
    _ALPHA_RANGE = [round(0.05 * (i + 1), 2) for i in range(int(1 / 0.05))]
    _SAVE_DIR = consts.BASE_DIR / consts.EVAL_FOLDER

    def __init__(self, args: ArgsProto, target: str, device: str | None = None):
        self._program_args = args
        self._pretrined = target
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model_builder = MergedModelBuilder(args.save)
        self._best_alpha: Optional[float] = None
        self._single_task_accuracies = self._load_single_task_accuracies()

    def _load_single_task_accuracies(self) -> Dict[str, Dict[str, float]]:
        json_path = (
            consts.BASE_DIR
            / consts.EVAL_FOLDER
            / consts.SINGLE_TASK_SAVE_FILE.format(suffix="_finetuned")
        )
        if not os.path.exists(str(json_path)):
            raise FileExistsError(
                "Json report for finetuned accuracies does not exist's."
            )
        with open(str(json_path), "r") as f:
            accuracies_json: List[utils.TaskAccuracyStat] = json.load(f)
        return {
            acc["dataset"]: {
                "train": acc["train"]["absolute"],
                "test": acc["test"]["absolute"],
                "validation": acc["validation"]["absolute"],
            }
            for acc in accuracies_json
        }

    def get_save_path(self, filename: str) -> str:
        os.makedirs(str(self._SAVE_DIR), exist_ok=True)
        return str(self._SAVE_DIR / filename)

    def set_best_alpha(self, alpha: float):
        self._best_alpha = alpha

    def find_best_alpha(self):
        best_alpha = 0.0
        best_avg_acc = 0.0
        single_task_accuracies = self._single_task_accuracies

        if len(single_task_accuracies) != len(self._model_builder.checkpoints):
            raise AssertionError(
                "Generated repot and accuracies from checkpoint are inconsistent"
            )

        for alpha in tqdm(self._ALPHA_RANGE, "Alpha picking"):
            accuracies = []
            for idx, ds in enumerate(self._model_builder.checkpoints.keys()):
                head = get_classification_head(self._program_args, ds)
                model = self._model_builder.build(head, self._pretrined, alpha)

                model.to(self._device)

                dataset = get_dataset(
                    ds + "Val",
                    preprocess=model.train_preprocess,
                    location=self._program_args.data_location,
                )
                data_loader = get_dataloader(
                    dataset,
                    args=self._program_args,
                    is_train=False,
                )
                loader_args = (
                    f"{ds}[{alpha:.2f}]",
                    idx + 1,
                    len(self._model_builder.checkpoints),
                )
                _, norm_acc = utils.evaluate_model(
                    model,
                    data_loader,
                    single_task_accuracies[ds]["validation"],
                    device=self._device,
                    loader_args=loader_args,
                )
                accuracies.append(norm_acc)

            print(f"\t{accuracies}")

            avg_norm_accuracy = sum(accuracies) / len(accuracies)
            if best_avg_acc < avg_norm_accuracy:
                best_alpha, best_avg_acc = (alpha, avg_norm_accuracy)

        print(f"Best alpha is: {best_alpha}")
        self._best_alpha = alpha
        return best_alpha

    def generate(
        self, path: str, fisher: bool = False
    ) -> List[Union[utils.TaskAccuracyStat, utils.TaskAccuracyStatsFisher]]:
        alpha = self._best_alpha or self.find_best_alpha()
        stat_gen = AccuracyStats(
            self._program_args, self._pretrined, self._program_args.save
        )
        self._program_args.st_alpha = alpha
        stats_result = stat_gen.generate(
            path, "merged", self._single_task_accuracies, fisher
        )
        return stats_result


if __name__ == "__main__":
    args = parse_arguments()
    checkpoints_dir = Path(args.save)
    stats_gen_multi_task = MultiTaskAccuracyStats(
        args, str(checkpoints_dir / consts.PRETRAINED_MODEL_NAME)
    )

    best_alpha = args.st_alpha or stats_gen_multi_task.find_best_alpha()
    stats_gen_multi_task.set_best_alpha(best_alpha)

    report_path = stats_gen_multi_task.get_save_path(
        consts.MULTI_TASK_SAVE_FILE.format(suffix=f"{best_alpha:.2f}".replace(".", "_"))
    )
    report = stats_gen_multi_task.generate(report_path, args.fisher)

    print("\nGenerated report for following datasets:")
    print(f"\t{str([stat['dataset'] for stat in report])}")
    print(f"\t Report saved to `{report_path}`")
