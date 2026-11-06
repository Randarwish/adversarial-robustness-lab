import torch
from torch.nn import functional as F


def pgd_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    step_size=2 / 255,
    steps=20,
):
    # Keep an unchanged reference for enforcing the attack limit.
    original = images.detach()

    # Start at a random point inside the permitted region.
    noise = torch.empty_like(original).uniform_(-epsilon, epsilon)
    attacked = (original + noise).clamp(0.0, 1.0)

    for _ in range(steps):
        # Start a fresh gradient calculation for this step.
        attacked = attacked.detach().requires_grad_(True)

        scores = model(attacked)
        loss = F.cross_entropy(scores, labels)

        gradient = torch.autograd.grad(loss, attacked)[0]

        # Take a small step toward higher loss.
        candidate = attacked.detach() + step_size * gradient.sign()

        # Project the change back into the permitted region.
        change = (candidate - original).clamp(-epsilon, epsilon)

        # Reconstruct the image and enforce valid pixel values.
        attacked = (original + change).clamp(0.0, 1.0)

    return attacked.detach()