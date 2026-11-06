from pathlib import Path

import torch
from torch.nn.functional import interpolate
from torchvision import datasets, transforms
from torchvision.utils import save_image

from attack_fgsm import fgsm_attack
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

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    epsilon = 8 / 255
    output_folder = Path("reports")
    output_folder.mkdir(exist_ok=True)

    for index in range(len(dataset)):
        image, label = dataset[index]

        # Add a batch dimension: (3, 32, 32) becomes (1, 3, 32, 32).
        image = image.unsqueeze(0).to(device)
        target = torch.tensor([label], device=device)

        with torch.no_grad():
            original_prediction = model(image).argmax(dim=1).item()

        # We want an example that was correct before the attack.
        if original_prediction != label:
            continue

        attacked = fgsm_attack(model, image, target, epsilon)

        with torch.no_grad():
            attacked_prediction = model(attacked).argmax(dim=1).item()

        # Keep searching if the attack did not change the prediction.
        if attacked_prediction == label:
            continue

        # Magnify the absolute pixel changes so we can inspect them.
        difference = (attacked - image).abs() / epsilon

        # Left: original. Middle: attacked. Right: magnified difference.
        comparison = torch.cat([image, attacked, difference], dim=0)

        # Enlarge the tiny images without smoothing their pixels.
        comparison = interpolate(
            comparison,
            scale_factor=8,
            mode="nearest",
        )

        save_image(
            comparison.cpu(),
            output_folder / "fgsm_example.png",
            nrow=3,
        )

        print("Test image index:", index)
        print("Correct category:", dataset.classes[label])
        print("Before attack:", dataset.classes[original_prediction])
        print("After attack:", dataset.classes[attacked_prediction])
        print(f"Maximum allowed change: {epsilon:.6f}")
        print(
            "Maximum actual change:",
            f"{(attacked - image).abs().max().item():.6f}",
        )
        print("Saved: reports/fgsm_example.png")
        return

    print("No successful attack found on an initially correct image.")


if __name__ == "__main__":
    main()