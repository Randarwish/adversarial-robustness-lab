import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attack_pgd import pgd_attack
from model import SmallClassifier


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path("checkpoints/adversarial_best.pt")
    report_path = Path("reports/defense_pgd.json")

    checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    checkpoint = torch.load(
        checkpoint_path,
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
    surviving_correct = 0
    total = 0
    largest_change = 0.0

    print("Device:", device)
    print("Defense checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint SHA256:", checkpoint_hash)
    print("Evaluating all 10,000 test images...", flush=True)

    for batch_number, (images, labels) in enumerate(
        loader, start=1
    ):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        # Input gradients must remain enabled during the attack.
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
        remains_correct = attacked_predictions == labels

        clean_correct += was_correct.sum().item()
        attacked_correct += remains_correct.sum().item()

        # An originally wrong prediction does not count as surviving.
        surviving_correct += (
            was_correct & remains_correct
        ).sum().item()

        total += labels.size(0)

        if batch_number % 10 == 0 or batch_number == len(loader):
            print(
                f"Batch {batch_number}/{len(loader)} complete",
                flush=True,
            )

    clean_accuracy = clean_correct / total
    attacked_accuracy = attacked_correct / total
    survival_accuracy = surviving_correct / total

    successful_attacks = clean_correct - surviving_correct
    success_rate = (
        successful_attacks / clean_correct
        if clean_correct > 0 else None
    )

    report = {
        "model": "adversarially fine-tuned classifier",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "starting_baseline_sha256": checkpoint[
            "configuration"
        ]["baseline_checkpoint_sha256"],
        "dataset": "CIFAR-10",
        "split": "test",
        "test_images": total,
        "attack": "PGD",
        "targeted": False,
        "norm": "Linf",
        "epsilon": epsilon,
        "step_size": step_size,
        "steps": steps,
        "random_start": True,
        "restarts": 1,
        "seed": 42,
        "batch_size": 128,
        "pixel_range": [0.0, 1.0],
        "clean_correct": clean_correct,
        "clean_accuracy": clean_accuracy,
        "correct_after_attack": attacked_correct,
        "accuracy_under_attack": attacked_accuracy,
        "correct_before_and_after_attack": surviving_correct,
        "pgd_survival_accuracy": survival_accuracy,
        "successful_attacks_on_initially_correct": successful_attacks,
        "attack_success_rate_on_initially_correct": success_rate,
        "largest_pixel_change": largest_change,
    }

    report_path.parent.mkdir(exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print(f"Clean test accuracy: {clean_accuracy:.2%}")
    print(f"Accuracy after PGD: {attacked_accuracy:.2%}")
    print(f"Accuracy surviving PGD: {survival_accuracy:.2%}")
    print(f"Correct before and after: {surviving_correct}/{total}")

    if success_rate is not None:
        print(
            "Attack success rate on initially correct images: "
            f"{success_rate:.2%}"
        )

    print(f"Maximum pixel change: {largest_change:.6f}")
    print("Saved:", report_path.as_posix())


if __name__ == "__main__":
    main()