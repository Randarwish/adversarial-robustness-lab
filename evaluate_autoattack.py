import hashlib
import json
from pathlib import Path

import torch
from autoattack import AutoAttack
from torchvision import datasets, transforms

from model import SmallClassifier


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path("checkpoints/baseline_best.pt")

    # Fingerprint the checkpoint so we can identify the exact model file.
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

    # Keep model weights fixed while allowing input gradients.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    sample_count = len(dataset)

    # Load the full test set as tensors.
    images = torch.stack([
        dataset[index][0] for index in range(sample_count)
    ]).to(device)

    labels = torch.tensor([
        dataset[index][1] for index in range(sample_count)
    ], dtype=torch.long, device=device)

    # Measure clean predictions in batches to limit working memory.
    clean_predictions = []

    with torch.no_grad():
        for start in range(0, sample_count, 128):
            batch = images[start:start + 128]
            predictions = model(batch).argmax(dim=1)
            clean_predictions.append(predictions)

    clean_predictions = torch.cat(clean_predictions)
    clean_correct = (clean_predictions == labels).sum().item()
    clean_accuracy = clean_correct / sample_count

    print("Device:", device)
    print("Test images:", sample_count)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint SHA256:", checkpoint_hash)
    print(f"Clean accuracy: {clean_accuracy:.2%}", flush=True)

    output_folder = Path("reports")
    output_folder.mkdir(exist_ok=True)

    epsilon = 8 / 255
    attack_batch_size = 50

    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=epsilon,
        version="standard",
        seed=42,
        device=str(device),
        log_path=str(output_folder / "autoattack_full.log"),
    )

    # Do not put this call inside torch.no_grad():
    # the attacks need gradients with respect to the images.
    attacked = adversary.run_standard_evaluation(
        images,
        labels,
        bs=attack_batch_size,
    )

    # Verify constraints and evaluate the returned images in batches.
    largest_change = 0.0
    surviving_correct = 0

    with torch.no_grad():
        for start in range(0, sample_count, 128):
            end = min(start + 128, sample_count)

            attacked_batch = attacked[start:end]
            original_batch = images[start:end]
            label_batch = labels[start:end]

            assert torch.isfinite(attacked_batch).all().item()
            assert attacked_batch.min().item() >= 0.0
            assert attacked_batch.max().item() <= 1.0

            change = (
                attacked_batch - original_batch
            ).abs().max().item()

            assert change <= epsilon + 1e-6
            largest_change = max(largest_change, change)

            predictions = model(attacked_batch).argmax(dim=1)

            survived = (
                (clean_predictions[start:end] == label_batch)
                & (predictions == label_batch)
            )

            surviving_correct += survived.sum().item()

    robust_accuracy = surviving_correct / sample_count
    successful_attacks = clean_correct - surviving_correct

    success_rate = (
        successful_attacks / clean_correct
        if clean_correct > 0 else None
    )

    report = {
        "attack": "AutoAttack",
        "version": "standard",
        "norm": "Linf",
        "epsilon": epsilon,
        "pixel_range": [0.0, 1.0],
        "seed": 42,
        "attack_batch_size": attack_batch_size,
        "dataset": "CIFAR-10",
        "split": "test",
        "subset": "all 10000 test images, indices 0-9999",
        "sample_count": sample_count,
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "clean_correct": clean_correct,
        "clean_accuracy": clean_accuracy,
        "correct_surviving_attack_suite": surviving_correct,
        "accuracy_surviving_attack_suite": robust_accuracy,
        "successful_attacks_on_initially_correct": successful_attacks,
        "attack_success_rate_on_initially_correct": success_rate,
        "largest_pixel_change": largest_change,
    }

    report_path = output_folder / "autoattack_full.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print(f"Clean accuracy: {clean_accuracy:.2%}")
    print(
        f"Accuracy surviving AutoAttack: {robust_accuracy:.2%}"
    )
    print(
        f"Correct after suite: {surviving_correct}/{sample_count}"
    )

    if success_rate is not None:
        print(
            "Attack success rate on initially correct images: "
            f"{success_rate:.2%}"
        )

    print(f"Maximum pixel change: {largest_change:.6f}")
    print(f"Saved: {report_path.as_posix()}")


if __name__ == "__main__":
    main()