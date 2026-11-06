import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


def main():
    # Convert images into numerical tensors with values from 0 to 1.
    transform = transforms.ToTensor()

    # Download the training images into the data folder.
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    # Collect 16 images into one batch.
    loader = DataLoader(
        training_data,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))

    print("Training images:", len(training_data))
    print("Categories:", training_data.classes)
    print("Image batch shape:", tuple(images.shape))
    print("Label batch shape:", tuple(labels.shape))
    print("Smallest pixel value:", images.min().item())
    print("Largest pixel value:", images.max().item())

    names = [training_data.classes[label.item()] for label in labels]
    print("Categories in this batch:", names)

    # Save the images as a grid with four images per row.
    save_image(images, "sample_images.png", nrow=4)
    print("Saved image grid: sample_images.png")


if __name__ == "__main__":
    main()