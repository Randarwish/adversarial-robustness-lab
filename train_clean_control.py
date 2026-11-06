import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import SmallClassifier
from train_adversarial import validate


def main():
    seed = 42
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    baseline_path = Path("checkpoints/baseline_best.pt")
    output_path = Path("checkpoints/clean_control_best.pt")
    history_path = Path("reports/clean_control_training.json")

    if output_path.exists() or history_path.exists():
        raise FileExistsError(
            "Control outputs already exist. Rename them before "
            "starting another experiment."
        )

    baseline_hash = hashlib.sha256(
        baseline_path.read_bytes()
    ).hexdigest()

    expected_baseline_hash = (
        "804dcdd45f32abade22306d8746c229ab"
        "3185b107cb7fb8086a10eb5c163801f"
    )

    if baseline_hash != expected_baseline_hash:
        raise ValueError(
            "The baseline checkpoint differs from the one used "
            "for the adversarial fine-tuning experiment."
        )

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
        "method": "clean-only fine-tuning from baseline",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": 128,
        "training_attack": None,
        "clean_loss_weight": 1.0,
        "attacked_loss_weight": 0.0,
        "validation_attack": "PGD",
        "validation_epsilon": 8 / 255,
        "validation_step_size": 2 / 255,
        "validation_attack_steps": 20,
        "validation_random_start": True,
        "validation_restarts": 1,
        "validation_seed": 12345,
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

    # Include the starting model as a candidate, as in the defense.
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
        model.train()

        total_loss = 0.0
        training_count = 0

        for batch_number, (images, labels) in enumerate(
            training_loader, start=1
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Train only on the original images.
            predictions = model(images)
            loss = loss_function(predictions, labels)

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

            print("Saved improved control checkpoint.", flush=True)

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