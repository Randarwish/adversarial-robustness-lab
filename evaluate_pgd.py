import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attack_pgd import pgd_attack
from model import SmallClassifier


def main():
    # Fix the random seed for the attack's random starting points.
    torch.manual_seed(42)

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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    epsilon = 8 / 255
    step_size = 2 / 255
    steps = 20

    clean_correct = 0
    attacked_correct = 0
    initially_correct = 0
    successful_attacks = 0
    total = 0
    largest_change = 0.0

    for batch_number, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        # Gradients must remain enabled while constructing the attack.
        attacked = pgd_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
        )

        change = (attacked - images).abs().max().item()
        largest_change = max(largest_change, change)

        assert torch.isfinite(attacked).all().item()
        assert attacked.min().item() >= 0.0
        assert attacked.max().item() <= 1.0
        assert change <= epsilon + 1e-6

        with torch.no_grad():
            attacked_predictions = model(attacked).argmax(dim=1)

        was_correct = clean_predictions == labels
        now_wrong = attacked_predictions != labels

        clean_correct += was_correct.sum().item()
        attacked_correct += (
            attacked_predictions == labels
        ).sum().item()

        initially_correct += was_correct.sum().item()
        successful_attacks += (was_correct & now_wrong).sum().item()
        total += labels.size(0)

        if batch_number % 10 == 0 or batch_number == len(loader):
            print(
                f"Batch {batch_number}/{len(loader)} complete",
                flush=True,
            )

    clean_accuracy = clean_correct / total
    attacked_accuracy = attacked_correct / total
    success_rate = (
        successful_attacks / initially_correct
        if initially_correct > 0 else None
    )

    report = {
        "attack": "PGD",
        "targeted": False,
        "norm": "Linf",
        "epsilon": epsilon,
        "step_size": step_size,
        "steps": steps,
        "random_start": True,
        "restarts": 1,
        "seed": 42,
        "pixel_range": [0.0, 1.0],
        "checkpoint": "checkpoints/baseline_best.pt",
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset": "CIFAR-10",
        "split": "test",
        "test_images": total,
        "clean_accuracy": clean_accuracy,
        "accuracy_under_attack": attacked_accuracy,
        "initially_correct": initially_correct,
        "successful_attacks_on_initially_correct": successful_attacks,
        "attack_success_rate_on_initially_correct": success_rate,
        "largest_pixel_change": largest_change,
    }

    Path("reports").mkdir(exist_ok=True)

    with open("reports/pgd.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"Clean accuracy: {clean_accuracy:.2%}")
    print(f"Accuracy under PGD: {attacked_accuracy:.2%}")

    if success_rate is not None:
        print(
            "Attack success rate on initially correct images: "
            f"{success_rate:.2%}"
        )

    print(f"Maximum actual pixel change: {largest_change:.6f}")
    print("Saved results: reports/pgd.json")


if __name__ == "__main__":
    main()