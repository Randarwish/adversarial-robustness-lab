import torch
from torch.nn import functional as F


def pgd_strong_attack(
    model,
    images,
    labels,
    epsilon=8 / 255,
    step_size=2 / 255,
    steps=40,
    restarts=5,
):
    """
    Untargeted PGD with multiple random starts.

    For each image:
    - Prefer a misclassified candidate over a correct candidate.
    - Among candidates with the same success status, prefer
      the one with higher cross-entropy loss.

    The caller must put the model in evaluation mode.
    Input pixels must lie within [0, 1].
    """

    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative.")

    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    if steps < 1 or restarts < 1:
        raise ValueError("steps and restarts must be at least 1.")

    original = images.detach()

    # Include the unchanged images as initial candidates.
    with torch.no_grad():
        original_scores = model(original)

        best_loss = F.cross_entropy(
            original_scores,
            labels,
            reduction="none",
        )

        best_success = (
            original_scores.argmax(dim=1) != labels
        )

    best_images = original.clone()

    for _ in range(restarts):
        noise = torch.empty_like(original).uniform_(
            -epsilon, epsilon
        )

        current = (original + noise).clamp(0.0, 1.0)

        # Evaluate the starting point and every resulting step.
        for step in range(steps + 1):
            current = current.detach().requires_grad_(True)

            scores = model(current)

            losses = F.cross_entropy(
                scores,
                labels,
                reduction="none",
            )

            with torch.no_grad():
                success = scores.argmax(dim=1) != labels
                candidate_loss = losses.detach()

                # A newly successful attack always replaces
                # a candidate that did not fool the model.
                newly_successful = success & ~best_success

                # Otherwise prefer higher loss among candidates
                # with the same success status.
                same_status = success == best_success
                higher_loss = candidate_loss > best_loss

                replace = newly_successful | (
                    same_status & higher_loss
                )

                best_images[replace] = current.detach()[replace]
                best_loss[replace] = candidate_loss[replace]
                best_success[replace] = success[replace]

            # The final candidate has been evaluated.
            if step == steps:
                break

            gradient = torch.autograd.grad(
                losses.mean(),
                current,
            )[0]

            with torch.no_grad():
                candidate = (
                    current + step_size * gradient.sign()
                )

                change = (candidate - original).clamp(
                    -epsilon, epsilon
                )

                current = (original + change).clamp(0.0, 1.0)

    return best_images.detach()