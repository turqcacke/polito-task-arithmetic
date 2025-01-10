from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
EVAL_FOLDER = "eval"
SINGLE_TASK_MODEL_TYPES = Literal["pretrained", "finetuned", "merged"]
SINGLE_TASK_SAVE_FILE = "single_task{suffix}.json"
PRETRAINED_MODEL_NAME = "pretrained_encoder.pt"
EXCLUDED_CHECKPOINTS = (PRETRAINED_MODEL_NAME,)
