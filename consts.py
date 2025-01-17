from pathlib import Path
from typing import Literal


BASE_DIR = Path(__file__).resolve().parent
EVAL_FOLDER = "eval"
DEFAULT_ALPHA = 1.0
SINGLE_TASK_MODEL_TYPES = Literal["pretrained", "finetuned", "merged"]
SINGLE_TASK_SAVE_FILE = "single_task{suffix}.json"
MULTI_TASK_SAVE_FILE = "multi_task_{suffix}.json"
PRETRAINED_MODEL_NAME = "pretrained_encoder.pt"
EXCLUDED_CHECKPOINTS = (PRETRAINED_MODEL_NAME,)
CHECKPOINTS_FOLDER = str(BASE_DIR / "checkpoints")