import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallClassifier


def fgsm_attack(model, images, labels, epsilon):
    # Create an independent input whose gradients we can calculate.
    inputs = images.detach().clone().requires_grad_(True)

    scores = model(inputs)
    loss = nn.functional.cross_entropy(scores, labels)

    # Calculate the gradient with respect to pixels.
    gradient = torch.autograd.grad(loss, inputs)[0]

    # Move pixels toward higher loss and keep valid pixel values.
    attacked = inputs + epsilon * gradient.sign()
    attacked = attacked.clamp(0.0, 1.0)

    return attacked.detach()


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

    # Freeze model weights. We only need gradients for input pixels.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

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

    # Fixed attack strengths for this experiment.
    epsilons = [0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255]
    results = []

    for epsilon in epsilons:
        correct = 0
        total = 0
        largest_change = 0.0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if epsilon == 0:
                attacked = images
            else:
                attacked = fgsm_attack(
                    model, images, labels, epsilon
                )

            change = (attacked - images).abs().max().item()
            largest_change = max(largest_change, change)

            # Check that the attack respects its pixel-change limit.
            assert change <= epsilon + 1e-6

            with torch.no_grad():
                predictions = model(attacked).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total

        results.append({
            "epsilon": epsilon,
            "epsilon_in_255_units": round(epsilon * 255),
            "accuracy_under_attack": accuracy,
            "correct_predictions": correct,
            "test_images": total,
            "largest_pixel_change": largest_change,
        })

        print(
            f"Epsilon: {epsilon * 255:.0f}/255 | "
            f"Accuracy: {accuracy:.2%} | "
            f"Correct: {correct}/{total}",
            flush=True,
        )

    report = {
        "attack": "FGSM",
        "targeted": False,
        "norm": "Linf",
        "pixel_range": [0.0, 1.0],
        "checkpoint": "checkpoints/baseline_best.pt",
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset": "CIFAR-10",
        "split": "test",
        "results": results,
    }

    Path("reports").mkdir(exist_ok=True)

    with open("reports/fgsm.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("Saved results: reports/fgsm.json")


if __name__ == "__main__":
    main()