from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
EVAL_FOLDER = "eval"
SINGLE_TASK_SAVE_FILE = "single_task.json"
PRETRAINED_MODEL_NAME = "pretrained_encoder.pt"
EXCLUDED_CHECKPOINTS = (PRETRAINED_MODEL_NAME,)
