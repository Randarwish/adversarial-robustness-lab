import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attack_pgd_strong import pgd_strong_attack
from model import SmallClassifier


def evaluate_checkpoint(name, checkpoint_path, loader, device):
    checkpoint_path = Path(checkpoint_path)

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

    # Reset attack randomness separately for each model.
    torch.manual_seed(42)

    epsilon = 8 / 255
    step_size = 2 / 255
    steps = 40
    restarts = 5

    clean_correct = 0
    surviving_correct = 0
    total = 0
    largest_change = 0.0

    print()
    print("Model:", name)
    print("Checkpoint:", checkpoint_path.as_posix())
    print("Checkpoint SHA256:", checkpoint_hash, flush=True)

    for batch_number, (images, labels) in enumerate(
        loader, start=1
    ):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        # Gradients must remain enabled for attack construction.
        attacked = pgd_strong_attack(
            model,
            images,
            labels,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            restarts=restarts,
        )

        change = (attacked - images).abs().max().item()

        assert torch.isfinite(attacked).all().item()
        assert attacked.min().item() >= 0.0
        assert attacked.max().item() <= 1.0
        assert change <= epsilon + 1e-6

        largest_change = max(largest_change, change)

        with torch.no_grad():
            attacked_predictions = model(attacked).argmax(dim=1)

        was_correct = clean_predictions == labels
        remains_correct = attacked_predictions == labels

        # The attack retains original errors and successful attacks.
        repaired = (~was_correct) & remains_correct

        if repaired.any().item():
            raise RuntimeError(
                "An original error was unexpectedly repaired. "
                "Check attack retention and model determinism."
            )

        clean_correct += was_correct.sum().item()
        surviving_correct += (
            was_correct & remains_correct
        ).sum().item()

        total += labels.size(0)

        if batch_number % 10 == 0 or batch_number == len(loader):
            print(
                f"{name} | Batch {batch_number}/{len(loader)}",
                flush=True,
            )

    clean_accuracy = clean_correct / total
    survival_accuracy = surviving_correct / total

    successful_attacks = clean_correct - surviving_correct
    success_rate = (
        successful_attacks / clean_correct
        if clean_correct > 0 else None
    )

    report = {
        "model": name,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset": "CIFAR-10",
        "split": "test",
        "test_images": total,
        "attack": "PGD with restarts and candidate retention",
        "targeted": False,
        "norm": "Linf",
        "epsilon": epsilon,
        "step_size": step_size,
        "steps_per_restart": steps,
        "restarts": restarts,
        "random_start": True,
        "includes_original_candidate": True,
        "retains_successful_intermediate_attacks": True,
        "candidate_selection": (
            "prefer misclassified candidates; "
            "then higher cross-entropy among equal success status"
        ),
        "seed": 42,
        "batch_size": loader.batch_size,
        "pixel_range": [0.0, 1.0],
        "clean_correct": clean_correct,
        "clean_accuracy": clean_accuracy,
        "correct_before_and_after_attack": surviving_correct,
        "pgd_survival_accuracy": survival_accuracy,
        "successful_attacks_on_initially_correct": successful_attacks,
        "attack_success_rate_on_initially_correct": success_rate,
        "largest_pixel_change": largest_change,
    }

    print(f"Clean accuracy: {clean_accuracy:.2%}")
    print(f"Strong PGD survival: {survival_accuracy:.2%}")
    print(f"Survivors: {surviving_correct}/{total}", flush=True)

    return report


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    models = [
        ("baseline", "checkpoints/baseline_best.pt"),
        ("clean_control", "checkpoints/clean_control_best.pt"),
        ("adversarial", "checkpoints/adversarial_best.pt"),
    ]

    # Check all required files before beginning the evaluations.
    for _, checkpoint_path in models:
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(checkpoint_path)

    output_folder = Path("reports")
    output_folder.mkdir(exist_ok=True)

    print("Device:", device)
    print("Test images per model:", len(dataset))
    print("Attack: 5 random starts, 40 steps each", flush=True)

    results = []

    for name, checkpoint_path in models:
        report = evaluate_checkpoint(
            name,
            checkpoint_path,
            loader,
            device,
        )

        results.append(report)

        # Save each completed model immediately.
        report_path = output_folder / f"pgd_strong_{name}.json"

        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("Saved:", report_path.as_posix(), flush=True)

    summary_path = output_folder / "pgd_strong_comparison.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"results": results}, file, indent=2)

    print()
    print("Completed comparison:")

    for report in results:
        print(
            f"{report['model']} | "
            f"Clean: {report['clean_accuracy']:.2%} | "
            f"Strong PGD survival: "
            f"{report['pgd_survival_accuracy']:.2%}"
        )

    print("Saved:", summary_path.as_posix())


if __name__ == "__main__":
    main()