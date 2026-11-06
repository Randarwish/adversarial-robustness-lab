import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallClassifier


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "checkpoints/baseline_best.pt",
        map_location=device,
        weights_only=True,
    )

    model = SmallClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = model(images)
            predictions = scores.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    results = {
        "checkpoint": "checkpoints/baseline_best.pt",
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset": "CIFAR-10",
        "split": "test",
        "test_images": total,
        "correct_predictions": correct,
        "clean_accuracy": accuracy,
    }

    Path("reports").mkdir(exist_ok=True)

    with open("reports/baseline.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test images:", total)
    print("Correct predictions:", correct)
    print(f"Clean test accuracy: {accuracy:.2%}")
    print("Saved results: reports/baseline.json")


if __name__ == "__main__":
    main()