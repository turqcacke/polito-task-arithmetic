#!/bin/bash

DATA_LOCATION="./data"
SAVE_DIR="./checkpoints"
BATCH_SIZE=32
LR=1e-4
WD=0.0
BALANCED="false"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --data-location)
      DATA_LOCATION="$2"
      shift 2
      ;;
    --save)
      SAVE_DIR="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --wd)
      WD="$2"
      shift 2
      ;;
    --balanced)
      BALANCED="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# 1) Finetune
echo "Starting finetuning..."
python finetune.py \
  --data-location=$DATA_LOCATION \
  --save=$SAVE_DIR \
  --batch-size=$BATCH_SIZE \
  --lr=$LR \
  --wd=$WD \
  --balance=$BALANCED

echo "Finetuning completed!"

# 2) Single Task evaluation
echo "Starting evaluation (Single Task)..."
python eval_single_task.py \
  --data-location=$DATA_LOCATION \
  --save=$SAVE_DIR

echo "Evaluation (Single Task) completed!"

# 3) Multi task avaluation
echo "Starting Task Addition evaluation..."
python eval_task_addition.py \
  --data-location=$DATA_LOCATION \
  --save=$SAVE_DIR

echo "Task Addition evaluation completed!"