from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import SmallClassifier


def main():
    # Set the random seed to make experiments easier to reproduce.
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Training on:", device)

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    training_data, validation_data = random_split(
        dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42),
    )

    training_loader = DataLoader(
        training_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    model = SmallClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_folder = Path("checkpoints")
    checkpoint_folder.mkdir(exist_ok=True)

    best_accuracy = -1.0
    epochs = 10

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        training_correct = 0
        training_count = 0

        for batch_number, (images, labels) in enumerate(
            training_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            scores = model(images)
            loss = loss_function(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            training_correct += (
                scores.argmax(dim=1) == labels
            ).sum().item()
            training_count += labels.size(0)

            if batch_number % 100 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_number}/{len(training_loader)}",
                    flush=True,
                )

        model.eval()

        validation_correct = 0
        validation_count = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                scores = model(images)
                validation_correct += (
                    scores.argmax(dim=1) == labels
                ).sum().item()
                validation_count += labels.size(0)

        training_loss = total_loss / training_count
        training_accuracy = training_correct / training_count
        validation_accuracy = validation_correct / validation_count

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {training_loss:.4f} | "
            f"Training: {training_accuracy:.2%} | "
            f"Validation: {validation_accuracy:.2%}",
            flush=True,
        )

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_accuracy": validation_accuracy,
                    "seed": 42,
                    "classes": dataset.classes,
                },
                checkpoint_folder / "baseline_best.pt",
            )
            print("Saved improved model.", flush=True)

    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()