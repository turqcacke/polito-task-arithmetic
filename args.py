import argparse
import consts
import os
from typing import Protocol
import torch
from typing import Optional, List, Literal, Protocol


class ArgsProto(Protocol):
    balance: bool
    fisher: bool
    st_model: Optional[consts.SINGLE_TASK_MODEL_TYPES]
    st_alpha: Optional[float]
    data_location: str
    model: str
    batch_size: int
    lr: float
    wd: float
    load: Optional[List[str]]
    save: Optional[str]
    cache_dir: Optional[str]
    openclip_cachedir: str
    device: str
    n_eval_points: int
    seed: Optional[int]
    stop_criterion: Literal["none", "fim", "valacc"]
    early_stop_patience: int


def parse_arguments() -> ArgsProto:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balance",
        type=bool,
        default=False,
        help="Flag whether balance train dataset or not",
    )
    parser.add_argument(
        "--st-model",
        choices=["pretrained", "finetuned", "merged"],
        default="finetuned",
        help="Which model is used for generating eval_single_task `json` report",
    )
    parser.add_argument(
        "--st-alpha",
        type=float,
        default=None,
        help="Alpha scaling (only for `prerained` and `finetuned` models)",
    )
    parser.add_argument(
        "--fisher",
        type=bool,
        default=False,
        help="Flag whether calculate fiher for multitask report or not",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser("~/data"),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B-32-quickgelu",  # New default model
        # default="ViT-B-32",
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",  # noqa: E501
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser("~/openclip-cachedir/open_clip"),
        help="Directory for caching models from OpenCLIP",
    )
    parser.add_argument(
        "--n-eval-points",
        type=int,
        default=21,
        help="Number of evaluation points used to find optimal coefficient in task arithmetic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--stop-criterion",
        type=str,
        default="none",
        choices=["none", "fim", "valacc"],
        help=(
            "Which stopping criterion to use: "
            "'none' => use the final epoch, "
            "'fim' => max FIM log-trace, "
            "'valacc' => max validation accuracy."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before stopping early.",
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
