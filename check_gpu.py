import torch


def main():
    print("PyTorch version:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)

    gpu_available = torch.cuda.is_available()
    print("Graphics card available:", gpu_available)

    if not gpu_available:
        print("Stop here: PyTorch cannot access the graphics card.")
        return

    print("Graphics card:", torch.cuda.get_device_name(0))

    # Create two matrices directly on the graphics card.
    first = torch.randn(100, 100, device="cuda")
    second = torch.randn(100, 100, device="cuda")

    # Multiply them using the graphics card.
    result = first @ second

    print("Result shape:", tuple(result.shape))
    print("Calculation device:", result.device)


if __name__ == "__main__":
    main()