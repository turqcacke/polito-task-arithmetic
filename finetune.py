import torch
import torch.nn as nn
from torch.optim import SGD
from args import parse_arguments
from datasets.common import get_dataloader
from datasets.registry import get_dataset
from datasets.common import maybe_dictionarize
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from utils import train_diag_fim_logtr
from tqdm import tqdm

def finetune_model(dataset_name, save_path, epochs = 0, lr=1e-4, batch_size=32):
    # Parse arguments and initialize device
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize pre-trained encoder and dataset-specific head
    encoder = ImageEncoder(args)
    head = get_classification_head(args, dataset_name + "Val")
    model = ImageClassifier(encoder, head).to(device)
    model.freeze_head()  # Freeze classification head

    # Load dataset
    train_dataset = get_dataset(
        dataset_name + "Val",
        preprocess=model.train_preprocess,
        location=args.data_location,
        batch_size=batch_size,
        num_workers=2
    )
    train_loader = get_dataloader(train_dataset, is_train=True, args=args)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.image_encoder.parameters(), lr=lr)

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

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Save fine-tuned encoder weights
    model.image_encoder.save(save_path)
    print(f"Fine-tuned weights saved to {save_path}")

if __name__ == "__main__":
    datasets = {
        # "DTD": 76,
        # "EuroSAT": 12,
        # "GTSRB": 11,
        "MNIST": 5,
        # "RESISC45": 15,
        # "SVHN": 4
    }
    save_directory = "./checkpoints/"

    for dataset_name, epochs in datasets.items():
        save_path = f"{save_directory}{dataset_name}_finetuned.pt"
        finetune_model(dataset_name, save_path, epochs)