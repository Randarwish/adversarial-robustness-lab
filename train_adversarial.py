import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from attack_pgd import pgd_attack
from model import SmallClassifier


def validate(model, loader, device):
    model.eval()

    clean_correct = 0
    attacked_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        # Validation uses a stronger attack than training.
        attacked = pgd_attack(
            model,
            images,
            labels,
            epsilon=8 / 255,
            step_size=2 / 255,
            steps=20,
        )

        with torch.no_grad():
            attacked_predictions = model(attacked).argmax(dim=1)

        clean_correct += (
            clean_predictions == labels
        ).sum().item()

        # Count only examples correct before and after the attack.
        attacked_correct += (
            (clean_predictions == labels)
            & (attacked_predictions == labels)
        ).sum().item()

        total += labels.size(0)

    return clean_correct / total, attacked_correct / total


def main():
    seed = 42
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    baseline_path = Path("checkpoints/baseline_best.pt")
    output_path = Path("checkpoints/adversarial_best.pt")
    history_path = Path("reports/adversarial_training.json")

    # Avoid accidentally replacing a completed defense experiment.
    if output_path.exists() or history_path.exists():
        raise FileExistsError(
            "Defense outputs already exist. Rename them before "
            "starting another experiment."
        )

    baseline_hash = hashlib.sha256(
        baseline_path.read_bytes()
    ).hexdigest()

    checkpoint = torch.load(
        baseline_path,
        map_location=device,
        weights_only=True,
    )

    model = SmallClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    # Reuse the baseline experiment's exact train/validation split.
    training_data, validation_data = random_split(
        dataset,
        [45000, 5000],
        generator=torch.Generator().manual_seed(seed),
    )

    training_loader = DataLoader(
        training_data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )

    epochs = 5
    learning_rate = 0.0001

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()

    configuration = {
        "method": "adversarial fine-tuning from baseline",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": 128,
        "epsilon": 8 / 255,
        "step_size": 2 / 255,
        "training_attack_steps": 7,
        "validation_attack_steps": 20,
        "random_start": True,
        "restarts": 1,
        "clean_loss_weight": 0.5,
        "attacked_loss_weight": 0.5,
        "seed": seed,
        "training_images": 45000,
        "validation_images": 5000,
        "selection": (
            "highest PGD validation survival accuracy, "
            "then highest clean validation accuracy"
        ),
        "baseline_checkpoint_sha256": baseline_hash,
    }

    output_path.parent.mkdir(exist_ok=True)
    history_path.parent.mkdir(exist_ok=True)

    print("Device:", device)
    print("Starting from:", baseline_path.as_posix())
    print("Baseline SHA256:", baseline_hash)
    print("Measuring starting validation performance...", flush=True)

    # Reset attack randomness so each validation starts from the same seed.
    torch.manual_seed(12345)

    clean_accuracy, attacked_accuracy = validate(
        model, validation_loader, device
    )

    print(
        f"Before fine-tuning | Clean: {clean_accuracy:.2%} | "
        f"PGD survival: {attacked_accuracy:.2%}",
        flush=True,
    )

    history = [{
        "epoch": 0,
        "clean_validation_accuracy": clean_accuracy,
        "pgd_validation_survival_accuracy": attacked_accuracy,
    }]

    best_score = (attacked_accuracy, clean_accuracy)

    # Include the starting checkpoint as a candidate.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "classes": dataset.classes,
            "configuration": configuration,
            "clean_validation_accuracy": clean_accuracy,
            "pgd_validation_survival_accuracy": attacked_accuracy,
        },
        output_path,
    )

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"configuration": configuration, "history": history},
            file,
            indent=2,
        )

    for epoch in range(1, epochs + 1):
        torch.manual_seed(seed + epoch)
        total_loss = 0.0
        training_count = 0

        for batch_number, (images, labels) in enumerate(
            training_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            # Construct difficult inputs using the current model.
            model.eval()
            attacked = pgd_attack(
                model,
                images,
                labels,
                epsilon=8 / 255,
                step_size=2 / 255,
                steps=7,
            )

            # Learn from both original and attacked images.
            model.train()
            optimizer.zero_grad()

            clean_loss = loss_function(model(images), labels)
            attacked_loss = loss_function(model(attacked), labels)

            loss = 0.5 * clean_loss + 0.5 * attacked_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            training_count += labels.size(0)

            if batch_number % 50 == 0:
                print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Batch {batch_number}/{len(training_loader)}",
                    flush=True,
                )

        print("Running validation...", flush=True)
        torch.manual_seed(12345)

        clean_accuracy, attacked_accuracy = validate(
            model, validation_loader, device
        )

        average_loss = total_loss / training_count

        history.append({
            "epoch": epoch,
            "training_loss": average_loss,
            "clean_validation_accuracy": clean_accuracy,
            "pgd_validation_survival_accuracy": attacked_accuracy,
        })

        print(
            f"Epoch {epoch}/{epochs} | Loss: {average_loss:.4f} | "
            f"Clean validation: {clean_accuracy:.2%} | "
            f"PGD validation survival: {attacked_accuracy:.2%}",
            flush=True,
        )

        score = (attacked_accuracy, clean_accuracy)

        if score > best_score:
            best_score = score

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "classes": dataset.classes,
                    "configuration": configuration,
                    "clean_validation_accuracy": clean_accuracy,
                    "pgd_validation_survival_accuracy": attacked_accuracy,
                },
                output_path,
            )

            print("Saved improved defense checkpoint.", flush=True)

        with history_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"configuration": configuration, "history": history},
                file,
                indent=2,
            )

    print("Finished.")
    print(f"Best PGD validation survival: {best_score[0]:.2%}")
    print(f"Clean validation at selection: {best_score[1]:.2%}")
    print("Checkpoint:", output_path.as_posix())
    print("History:", history_path.as_posix())


if __name__ == "__main__":
    main()