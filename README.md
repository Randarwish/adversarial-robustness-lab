# Adversarial Robustness Lab

A hands-on experiment investigating how gradient-based adversarial
perturbations affect an image classifier.

## Current experiment

- Dataset: CIFAR-10.
- Model: a small convolutional neural network with 545,098 parameters.
- Training: 45,000 images, with 5,000 separate validation images.
- Evaluation: all 10,000 test images.
- Checkpoint: selected using the highest validation accuracy.
- Attack: untargeted, white-box Fast Gradient Sign Method (FGSM).
- Inputs: pixel values between 0 and 1.
- Constraint: maximum absolute change per pixel channel.

## Measured results

| Maximum pixel change | Test accuracy |
|---|---:|
| No attack | 70.10% |
| 1/255 | 50.69% |
| 2/255 | 34.07% |
| 4/255 | 13.23% |
| 8/255 | 1.47% |

The baseline model is vulnerable to FGSM under these settings.
These results describe one training run and do not establish
robustness against other attacks.

## Example

An initially correct prediction changed from cat to dog at 8/255.

Left: original image.
Middle: attacked image.
Right: absolute pixel differences magnified for visibility.

![FGSM example](reports/fgsm_example.png)

This example was selected because the attack succeeded.
The table above reports performance across the full test dataset.

## Run locally

The experiment was run on Windows using Python 3.11.9,
PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128,
and an NVIDIA GeForce RTX 4050 Laptop GPU.

Create a Python environment and install the packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128