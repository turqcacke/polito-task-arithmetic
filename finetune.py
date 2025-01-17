import os
import json
import torch
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from args import parse_arguments, ArgsProto
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr, evaluate_model
from consts import CHECKPOINTS_FOLDER, BASE_DIR, EVAL_FOLDER
from tqdm import tqdm


def save_pretrained_encoder(encoder, save_dir):
    """Save the pre-trained encoder before fine-tuning."""
    os.makedirs(save_dir, exist_ok=True)
    pretrained_path = os.path.join(save_dir, "pretrained_encoder.pt")
    encoder.save(pretrained_path)


def finetune_model(
    dataset_name: str,
    args: ArgsProto,
    save_path: str,
    epochs=0,
    lr=1e-4,
    batch_size=32,
    balance_ds=False,
):
    if epochs <= 0:
        raise ValueError("The number of epochs must be greater than 0.")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize pre-trained encoder and dataset-specific head
    encoder = ImageEncoder(args)
    save_pretrained_encoder(encoder, "./checkpoints/")

    head = get_classification_head(args, dataset_name)
    model = ImageClassifier(encoder, head).to(device)
    model.freeze_head()

    # Load training datasets
    train_dataset = get_dataset(
        dataset_name + "Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=batch_size,
        num_workers=2,
        balance=balance_ds,
    )

    train_loader = get_dataloader(train_dataset, is_train=True, args=args)
    val_loader = get_dataloader(train_dataset, is_train=False, args=args)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.image_encoder.parameters(), lr=lr, weight_decay=args.wd)

    best_metric_value = float("-inf")
    best_epoch = 0
    metrics = []

    print(f"Starting fine-tuning for {dataset_name}...")

    # Fine-tuning loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data = maybe_dictionarize(batch)
            images = data["images"].to(device)
            labels = data["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}")

        # Metric computation
        current_metric = None
        if args.stop_criterion == "fim":
            current_metric = train_diag_fim_logtr(
                args, model, dataset_name, samples_nr=args.n_eval_points or 2000
            )
            print(f"   FIM log-trace = {current_metric:.4f}")
        elif args.stop_criterion == "valacc":
            val_acc, _ = evaluate_model(model, val_loader, device=device)
            current_metric = val_acc
            print(f"   Validation Accuracy: {val_acc:.4f}")

        # Check for improvements and save checkpoint
        if current_metric is not None and current_metric > best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch + 1
            model.image_encoder.save(save_path)
            print(f"   [Best Checkpoint Saved]: {save_path}")

        metrics.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "metric": current_metric if current_metric is not None else "NA",
            }
        )

    if args.stop_criterion == "none":
        # Save final checkpoint
        final_checkpoint_path = save_path
        model.image_encoder.save(final_checkpoint_path)
        print(f"Final checkpoint saved at {final_checkpoint_path}.")

    # Save training metrics
    with open(str(BASE_DIR / EVAL_FOLDER / (dataset_name + "_metrics.json")), "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"Training completed for {dataset_name}. Best epoch: {best_epoch}, Best metric: {best_metric_value:.4f}"
    )


if __name__ == "__main__":
    args = parse_arguments()
    datasets = {
        # "DTD": 76,
        # "EuroSAT": 12,
        # "GTSRB": 11,
        # "MNIST": 5,
        # "RESISC45": 15,
        "SVHN": 4,
    }
    for dataset_name, epochs in datasets.items():
        save_path = CHECKPOINTS_FOLDER / f"{dataset_name}_finetuned.pt"
        finetune_model(
            dataset_name,
            args,
            str(save_path),
            epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            balance_ds=args.balance,
        )
