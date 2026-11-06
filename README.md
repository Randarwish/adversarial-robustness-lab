# Adversarial Robustness Lab

An experiment comparing a CIFAR-10 image classifier before and
after adversarial fine-tuning.

## Main finding

Five epochs of adversarial fine-tuning increased test accuracy
under the standard AutoAttack suite from 0.06% to 6.72% at an
8/255 pixel-change limit.

Clean test accuracy decreased from 70.29% to 63.07%.

The defense improved resistance to the evaluated attacks, but
substantial vulnerability remains. These results are empirical
measurements, not security guarantees.

## Test results

| Evaluation | Baseline | Adversarially fine-tuned |
|---|---:|---:|
| Clean accuracy | 70.29% | 63.07% |
| PGD accuracy at 8/255 | 0.07% | 11.52% |
| AutoAttack accuracy at 8/255 | 0.06% | 6.72% |

All evaluations above use the full 10,000-image test split.

The defense increased AutoAttack accuracy by 6.66 percentage points
and reduced clean accuracy by 7.22 percentage points.

After AutoAttack, 672 images remained correctly classified by the
defense, compared with 6 by the baseline.

AutoAttack fooled 89.35% of the defense's initially correct
predictions.

### Baseline FGSM results

| Maximum pixel-channel change | Test accuracy |
|---|---:|
| No attack | 70.29% |
| 1/255 | 50.18% |
| 2/255 | 33.98% |
| 4/255 | 13.27% |
| 8/255 | 1.54% |

These measurements supersede the earlier baseline results.

## Model and data

- Dataset: CIFAR-10.
- Architecture: a small convolutional neural network.
- Trainable parameters: 545,098.
- Training images: 45,000.
- Validation images: 5,000.
- Test images: 10,000.
- Input shape: 3 color channels, 32 pixels high, 32 pixels wide.
- Pixel values: [0, 1].
- Train/validation split seed: 42.

The baseline checkpoint is selected by clean validation accuracy.

The defense checkpoint is selected by PGD validation survival
accuracy, breaking ties using clean validation accuracy.
The starting checkpoint is also included as a selection candidate.

Training images update model parameters. Validation images select
checkpoints. The separate test split measures final performance.

## Threat model

During evaluation, model weights remain fixed while attacks modify
image pixels.

The perturbation limit is the maximum absolute change to any pixel
channel, also called the Linf constraint. At 8/255, that limit is
approximately 0.031373 on the [0, 1] input scale.

FGSM and our custom PGD attack are untargeted, white-box attacks
using the correct labels.

AutoAttack's standard suite combines gradient-based and query-based
methods and includes targeted components.

## Attack configurations

### Fast Gradient Sign Method

- One gradient-based step.
- Cross-entropy loss.
- Epsilon: 0, 1/255, 2/255, 4/255, and 8/255.
- Resulting pixel values clipped to [0, 1].

### Projected Gradient Descent

- Epsilon: 8/255.
- Step size: 2/255.
- Steps: 20.
- Cross-entropy loss.
- One random starting point per image.
- No additional restarts.
- Test evaluation seed: 42.
- Batch size: 128.
- Projection into the permitted region after every step.

The defense report distinguishes accuracy after the attack from
survival accuracy: the fraction of all images classified correctly
both before and after the attack.

Both defense measurements were 11.52% in this run.

### AutoAttack

Official implementation:
https://github.com/fra31/auto-attack

- Version: standard.
- Components: APGD-CE, APGD-T, FAB-T, and Square Attack.
- Epsilon: 8/255.
- Seed: 42.
- Attack batch size: 50.
- Evaluation size: all 10,000 test images.
- Source revision: a39220048b3c9f2cca9a4d3a54604793c68eca7e.

Later components attack images that remain correctly classified
after earlier components.

Defense results by stage:

| Stage | Survival accuracy |
|---|---:|
| Clean images | 63.07% |
| After APGD-CE | 9.91% |
| After APGD-T | 6.73% |
| After FAB-T | 6.72% |
| After Square Attack | 6.72% |

The suite found failures missed by our 20-step PGD evaluation.
Surviving the suite does not prove that an image is robust.

## Adversarial fine-tuning

The defense starts from the trained baseline checkpoint.

For each training batch, PGD generates attacked images using the
current model. Model parameters are then updated using an equal
mixture of clean and attacked-image cross-entropy losses.

Configuration:

- Epochs: 5.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 128.
- Training attack: 7-step PGD.
- Training epsilon: 8/255.
- Training step size: 2/255.
- One random starting point per image.
- Loss weights: 50% clean, 50% attacked.
- Validation attack: 20-step PGD at the same epsilon.

## Example

An initially correct baseline prediction changed from cat to dog
under FGSM at 8/255.

Left: original image.
Middle: attacked image.
Right: magnified absolute pixel differences.

![FGSM example](reports/fgsm_example.png)

This example was selected because the attack succeeded.
The numerical comparison uses the full test split.

## Run locally

Recorded environment:

- Windows.
- Python 3.11.9.
- PyTorch 2.11.0+cu128.
- Torchvision 0.26.0+cu128.
- NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB graphics memory.

Git must be installed to install AutoAttack from its pinned source
revision. The requirements target a compatible NVIDIA environment.

From the project folder, create an environment and install:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements pin PyTorch, Torchvision, and the exact AutoAttack
revision used in this experiment. Supporting dependencies are not
fully locked.

### Baseline

```powershell
.\.venv\Scripts\python.exe check_gpu.py
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe attack_fgsm.py
.\.venv\Scripts\python.exe visualize_attack.py
.\.venv\Scripts\python.exe evaluate_pgd.py
.\.venv\Scripts\python.exe evaluate_autoattack.py
```

### Defense

```powershell
.\.venv\Scripts\python.exe train_adversarial.py
.\.venv\Scripts\python.exe evaluate_defense.py
.\.venv\Scripts\python.exe evaluate_defense_autoattack.py
```

The dataset downloads during the first inspection run.
Training creates checkpoints, which are excluded from Git.

The defense training script stops if its checkpoint or history
already exists, preventing accidental replacement.

If a checkpoint changes, rerun all evaluations that use it.
Exact training results may vary across environments.

## Checkpoint identities

Evaluation reports record SHA256 fingerprints identifying the
model files used.

Baseline:

```text
804dcdd45f32abade22306d8746c229ab3185b107cb7fb8086a10eb5c163801f
```

Defense:

```text
6c9503043e6f9d8b5108277ba92684e4f131e83c92d9d9a129e3f96d43d4fdfc
```

Fingerprints do not guarantee deterministic training or model security.

## Project files

| File | Purpose |
|---|---|
| `requirements.txt` | Pin primary packages and AutoAttack revision |
| `check_gpu.py` | Verify graphics-card access |
| `inspect_data.py` | Download and inspect CIFAR-10 |
| `model.py` | Define the classifier |
| `train.py` | Train the baseline |
| `evaluate.py` | Evaluate clean baseline accuracy |
| `attack_fgsm.py` | Implement and evaluate FGSM |
| `visualize_attack.py` | Save a baseline attack example |
| `attack_pgd.py` | Implement PGD |
| `evaluate_pgd.py` | Evaluate baseline PGD performance |
| `evaluate_autoattack.py` | Evaluate baseline AutoAttack performance |
| `train_adversarial.py` | Fine-tune with clean and attacked images |
| `evaluate_defense.py` | Evaluate defense clean and PGD performance |
| `evaluate_defense_autoattack.py` | Evaluate defense with AutoAttack |

## Reports

| File | Contents |
|---|---|
| `reports/baseline.json` | Clean baseline results |
| `reports/fgsm.json` | Baseline FGSM results |
| `reports/pgd.json` | Baseline PGD results |
| `reports/fgsm_example.png` | Selected attack example |
| `reports/autoattack_trial.json` | Preliminary 100-image results |
| `reports/autoattack_trial.log` | Preliminary evaluation log |
| `reports/autoattack_full.json` | Full baseline AutoAttack results |
| `reports/autoattack_full.log` | Full baseline AutoAttack log |
| `reports/adversarial_training.json` | Defense configuration and validation history |
| `reports/defense_pgd.json` | Defense clean and PGD results |
| `reports/defense_autoattack.json` | Defense AutoAttack results |
| `reports/defense_autoattack.log` | Defense AutoAttack log |

The preliminary trial is not included in the full-test comparison.

## Limitations

- One architecture and one baseline/defense checkpoint pair.
- Training has not been repeated across multiple seeds.
- The defense receives additional training; there is no matched
  clean-only fine-tuning control.
- Custom PGD uses one random start and its final iterate rather
  than retaining successful intermediate attacks.
- Supporting dependencies are not fully locked.
- Original checkpoints are not distributed.
- Results concern bounded pixel perturbations on CIFAR-10.
- No certified robustness or deployment-security guarantee is provided.

## Next experiments

- Add a matched clean-only fine-tuning control.
- Strengthen custom PGD evaluation with multiple restarts and
  retention of successful intermediate attacks.
- Repeat training across multiple seeds.