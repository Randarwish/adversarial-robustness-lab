# Adversarial Robustness Lab

A hands-on experiment investigating how gradient-based adversarial
perturbations affect an image classifier.

## Current experiment

- Dataset: CIFAR-10.
- Model: a small convolutional neural network with 545,098 parameters.
- Training: 45,000 images, with 5,000 separate validation images.
- Evaluation: all 10,000 test images.
- Checkpoint: selected using the highest validation accuracy.
- Attacks: untargeted, white-box attacks using the correct labels.
- Inputs: pixel values between 0 and 1.
- Constraint: maximum absolute change per pixel channel.

During an attack, the model's weights remain fixed.
The attack modifies the input image to increase classification loss.

## Measured results

| Evaluation | Maximum pixel change | Test accuracy |
|---|---:|---:|
| Clean images | 0 | 70.29% |
| Fast Gradient Sign Method | 1/255 | 50.18% |
| Fast Gradient Sign Method | 2/255 | 33.98% |
| Fast Gradient Sign Method | 4/255 | 13.27% |
| Fast Gradient Sign Method | 8/255 | 1.54% |
| Projected Gradient Descent | 8/255 | 0.07% |

All results above were measured using the same current checkpoint.
They supersede the earlier measurements.

At the same 8/255 limit, Fast Gradient Sign Method leaves 154
correct predictions, while Projected Gradient Descent leaves 7.

These results demonstrate vulnerability under the tested settings.
They describe one model checkpoint and do not establish robustness
against every possible attack.

## Attack configurations

### Fast Gradient Sign Method

- One gradient-based step.
- Evaluated at 0, 1/255, 2/255, 4/255, and 8/255.
- Pixel values clipped to the interval [0, 1].

### Projected Gradient Descent

- Maximum pixel-channel change: 8/255.
- Step size: 2/255.
- Steps: 20.
- One random starting point per image, with no additional restarts.
- Random seed: 42.
- Perturbation projected into the permitted region after every step.
- Pixel values clipped to the interval [0, 1].
- Attack success rate among initially correct images: 99.90%.

Attack success rate measures the fraction of originally correct
predictions that become incorrect after the attack.

## Example

An initially correct prediction changed from cat to dog under
Fast Gradient Sign Method at 8/255.

From left to right:

1. Original image.
2. Attacked image.
3. Absolute pixel differences magnified for visibility.

![FGSM example](reports/fgsm_example.png)

This example was selected because the attack succeeded.
The results table reports performance across all 10,000 test images.

## Run locally

The experiment was run on Windows using Python 3.11.9,
PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128,
and an NVIDIA GeForce RTX 4050 Laptop GPU.

The installation command below uses the CUDA 12.8 package index
for a compatible NVIDIA environment.

Create an environment and install the packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

Run the scripts in order:

```powershell
.\.venv\Scripts\python.exe check_gpu.py
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe attack_fgsm.py
.\.venv\Scripts\python.exe visualize_attack.py
.\.venv\Scripts\python.exe evaluate_pgd.py
```

The dataset downloads during the first inspection run.
Training creates the checkpoint needed by the evaluation scripts.

Use the same checkpoint for all evaluations in a comparison.
If you retrain, rerun every evaluation and refresh the example image.

Exact results may vary across environments.

## Project files

| File | Purpose |
|---|---|
| `check_gpu.py` | Verify graphics-card access |
| `inspect_data.py` | Load and inspect the dataset |
| `model.py` | Define the classifier |
| `train.py` | Train and save the best validation checkpoint |
| `evaluate.py` | Measure clean test accuracy |
| `attack_fgsm.py` | Implement and evaluate the single-step attack |
| `visualize_attack.py` | Save a selected successful attack example |
| `attack_pgd.py` | Implement the repeated-step attack |
| `evaluate_pgd.py` | Evaluate the repeated-step attack |

## Saved outputs

- `reports/baseline.json`: clean test evaluation.
- `reports/fgsm.json`: single-step attack results.
- `reports/pgd.json`: repeated-step attack results and configuration.
- `reports/fgsm_example.png`: selected attack example.

Datasets, local environments, and model checkpoints are excluded
from Git. Reproducing the experiment requires training a checkpoint.

## Next experiments

- AutoAttack evaluation.
- Adversarial training.
- Comparison of clean accuracy and accuracy under attack.