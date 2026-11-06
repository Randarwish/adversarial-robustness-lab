import torch
from torch import nn


class SmallClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SmallClassifier().to(device)

    # Four artificial images, used only to check the model's shape.
    images = torch.rand(4, 3, 32, 32, device=device)

    with torch.no_grad():
        scores = model(images)

    print("Device:", device)
    print("Input shape:", tuple(images.shape))
    print("Output shape:", tuple(scores.shape))

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    print("Trainable parameters:", parameter_count)


if __name__ == "__main__":
    main()