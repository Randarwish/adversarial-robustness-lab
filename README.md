# Adversarial Robustness Lab

A CIFAR-10 experiment comparing an image classifier, clean-only
fine-tuning, and adversarial fine-tuning under adversarial attacks.

## Main finding

Adversarial fine-tuning increased test accuracy under the standard
AutoAttack suite from 0.06% to 6.72%, while clean accuracy decreased
from 70.29% to 63.07%.

Clean-only fine-tuning improved clean accuracy to 72.05% but did
not produce a comparable improvement in attack resistance.

This supports the benefit of adversarial training in this experiment.
It does not establish a general security guarantee.

## Final test comparison

All attacks below use an 8/255 maximum pixel-channel change.
All evaluations cover the same 10,000 test images.

| Model | Clean accuracy | Multi-start PGD survival | AutoAttack survival |
|---|---:|---:|---:|
| Original baseline | 70.29% | 0.07% | 0.06% |
| Clean-only control | 72.05% | 0.06% | 0.03% |
| Adversarially fine-tuned | 63.07% | 10.46% | 6.72% |

Compared with the baseline, adversarial fine-tuning increased
AutoAttack survival by 6.66 percentage points and reduced clean
accuracy by 7.22 percentage points.

AutoAttack left 6 baseline images, 3 control images, and 672 defense
images correctly classified.

The difference between three and six survivors should not be
interpreted as an established effect across repeated experiments.

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

Training images update model parameters.
Validation images select checkpoints.
The separate test split measures selected models.

## Experiment design

### Baseline

The baseline is trained for 10 epochs using Adam with a learning
rate of 0.001 and batch size 128.

The checkpoint is selected by highest clean validation accuracy.

### Fine-tuning comparison

Both fine-tuning runs start from the same baseline checkpoint
and complete five additional epochs.

Shared settings:

- Architecture and data split.
- Batch size: 128.
- Optimizer: Adam, initialized separately for each run.
- Learning rate: 0.0001.
- Training-data shuffle generator seed: 42.
- Validation procedure.
- Checkpoint-selection rule.

Selection prioritizes 20-step PGD validation survival, breaking ties
using clean validation accuracy. The starting checkpoint is also
a selection candidate.

The selected control is from fine-tuning epoch 1.
The selected defense is from fine-tuning epoch 5.

Training budgets and selection policies match, but selected
checkpoint durations differ. Computation cost is not matched.

### Clean-only control

Training uses original images only:

```text
loss = clean cross-entropy
```

Attacked validation images select checkpoints but do not update
model parameters.

### Adversarial fine-tuning

Attacked training images are regenerated using the current model:

```text
loss = 0.5 × clean cross-entropy
     + 0.5 × attacked cross-entropy
```

Training attack:

- PGD with 7 steps.
- Epsilon: 8/255.
- Step size: 2/255.
- One random start per image.

## Threat model

Evaluation attacks keep model weights fixed and modify image pixels.

The constraint is the maximum absolute change to any pixel channel,
also called the Linf constraint. At 8/255, that limit is approximately
0.031373 on the [0, 1] scale.

Our FGSM and PGD attacks are untargeted white-box attacks using the
correct labels.

AutoAttack combines gradient-based and query-based methods and
includes targeted components.

Survival accuracy counts images classified correctly both before
and after an attack, divided by the full evaluation sample count.
Attack success rate instead uses initially correct images as its
denominator.

## Attack configurations

### Fast Gradient Sign Method

- One gradient-based step.
- Cross-entropy loss.
- Pixel values clipped to [0, 1].

Baseline results:

| Maximum pixel-channel change | Test accuracy |
|---|---:|
| No attack | 70.29% |
| 1/255 | 50.18% |
| 2/255 | 33.98% |
| 4/255 | 13.27% |
| 8/255 | 1.54% |

These values supersede earlier baseline measurements.
FGSM was not separately evaluated on the fine-tuned models.

### Original PGD

- Epsilon: 8/255.
- Step size: 2/255.
- Steps: 20.
- One random start.
- Cross-entropy loss.
- Final iterate returned.
- Test seed: 42.
- Batch size: 128.

| Model | Accuracy after original PGD |
|---|---:|
| Baseline | 0.07% |
| Adversarially fine-tuned | 11.52% |

The defense's survival accuracy was also 11.52%.
The control did not receive this separate original-PGD test.

### Multi-start PGD with candidate retention

- Epsilon: 8/255.
- Step size: 2/255.
- Steps per random start: 40.
- Random starts per image: 5.
- Cross-entropy loss.
- Seed: 42.
- Batch size: 128.
- Original image included as a candidate.
- Starting points and every intermediate iterate evaluated.
- Successful attacks retained separately for each image.
- Among candidates with equal success status, higher loss preferred.

| Model | Survival accuracy | Survivors |
|---|---:|---:|
| Baseline | 0.07% | 7 |
| Clean-only control | 0.06% | 6 |
| Adversarially fine-tuned | 10.46% | 1,046 |

For the defense, this attack found 106 more failures than the
original PGD evaluation.

These are separate attack evaluations, not a per-image union of
failures across all evaluated attacks.

### AutoAttack

Official implementation:
https://github.com/fra31/auto-attack

- Version: standard.
- Components: APGD-CE, APGD-T, FAB-T, and Square Attack.
- Epsilon: 8/255.
- Seed: 42.
- Attack batch size: 50.
- Source revision: a39220048b3c9f2cca9a4d3a54604793c68eca7e.

Later components attack images that survive earlier components.

Defense survival by stage:

| Stage | Survival accuracy |
|---|---:|
| Clean images | 63.07% |
| After APGD-CE | 9.91% |
| After APGD-T | 6.73% |
| After FAB-T | 6.72% |
| After Square Attack | 6.72% |

Attack success among initially correct images:

| Model | Attack success rate |
|---|---:|
| Baseline | 99.91% |
| Clean-only control | 99.96% |
| Adversarially fine-tuned | 89.35% |

Surviving an attack does not prove that an image is robust.

## Example attack

An initially correct baseline prediction changed from cat to dog
under FGSM at 8/255.

Left: original image.
Middle: attacked image.
Right: magnified absolute pixel differences.

![FGSM example](reports/fgsm_example.png)

This is a selected successful example.
The numerical comparisons use the full test split.

## Run locally

Recorded environment:

- Windows.
- Python 3.11.9.
- PyTorch 2.11.0+cu128.
- Torchvision 0.26.0+cu128.
- NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB graphics memory.

Git is required to install AutoAttack from its pinned source.
The requirements target a compatible NVIDIA environment.

From the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Primary packages and the AutoAttack revision are pinned.
Supporting dependencies are not fully locked.

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

### Adversarial fine-tuning

```powershell
.\.venv\Scripts\python.exe train_adversarial.py
.\.venv\Scripts\python.exe evaluate_defense.py
.\.venv\Scripts\python.exe evaluate_defense_autoattack.py
```

### Clean-only control

```powershell
.\.venv\Scripts\python.exe train_clean_control.py
.\.venv\Scripts\python.exe evaluate_clean_control.py
```

The control imports validation from `train_adversarial.py`.
Importing it does not run defense training.

### Multi-start PGD comparison

After all three checkpoints exist:

```powershell
.\.venv\Scripts\python.exe evaluate_pgd_strong.py
```

The dataset downloads during the first inspection run.
Training creates checkpoints, which are excluded from Git.

Fine-tuning scripts stop if their output checkpoint or training
history already exists.

The control checks the original baseline fingerprint. Reproducing
with a newly trained baseline requires explicitly updating that
experiment check and reevaluating all models.

Exact results may vary across environments.
If a checkpoint changes, rerun its evaluations.

## Checkpoint identities

SHA256 fingerprints identify the evaluated model files.

Baseline:

```text
804dcdd45f32abade22306d8746c229ab3185b107cb7fb8086a10eb5c163801f
```

Clean-only control:

```text
702b338368e153275ddc23dc03b98a1ee9c1e795c14c5a66f806574d9f29da91
```

Adversarially fine-tuned:

```text
6c9503043e6f9d8b5108277ba92684e4f131e83c92d9d9a129e3f96d43d4fdfc
```

Fingerprints do not guarantee deterministic training or model security.

## Project files

| File | Purpose |
|---|---|
| `requirements.txt` | Pin primary dependencies |
| `check_gpu.py` | Verify graphics-card access |
| `inspect_data.py` | Download and inspect CIFAR-10 |
| `model.py` | Define the classifier |
| `train.py` | Train the baseline |
| `evaluate.py` | Evaluate clean baseline accuracy |
| `attack_fgsm.py` | Implement and evaluate FGSM |
| `visualize_attack.py` | Save an attack example |
| `attack_pgd.py` | Implement original PGD |
| `evaluate_pgd.py` | Evaluate baseline original PGD |
| `evaluate_autoattack.py` | Evaluate baseline AutoAttack |
| `train_adversarial.py` | Train the defense and define validation |
| `evaluate_defense.py` | Evaluate defense clean and PGD accuracy |
| `evaluate_defense_autoattack.py` | Evaluate defense AutoAttack |
| `train_clean_control.py` | Train the clean-only control |
| `evaluate_clean_control.py` | Evaluate control AutoAttack |
| `attack_pgd_strong.py` | Implement multi-start PGD with retention |
| `evaluate_pgd_strong.py` | Evaluate all three models with stronger PGD |

## Saved reports

All reports are in `reports/`.

- `baseline.json`: clean baseline evaluation.
- `fgsm.json`: baseline FGSM evaluation.
- `pgd.json`: baseline original PGD evaluation.
- `fgsm_example.png`: selected attack example.
- `autoattack_trial.json` and `.log`: preliminary 100-image trial.
- `autoattack_full.json` and `.log`: full baseline AutoAttack evaluation.
- `adversarial_training.json`: defense configuration and training history.
- `defense_pgd.json`: defense clean and original PGD results.
- `defense_autoattack.json` and `.log`: defense AutoAttack results.
- `clean_control_training.json`: control configuration and history.
- `clean_control_autoattack.json` and `.log`: control AutoAttack results.
- `pgd_strong_baseline.json`: stronger PGD baseline results.
- `pgd_strong_clean_control.json`: stronger PGD control results.
- `pgd_strong_adversarial.json`: stronger PGD defense results.
- `pgd_strong_comparison.json`: combined stronger PGD comparison.

The preliminary trial is excluded from the main comparison.

## Limitations

- One architecture and one run per fine-tuning method.
- No repeated training across multiple seeds.
- Selected fine-tuning checkpoints have different training durations.
- Training computation cost is not matched.
- Original PGD has weaker search and retention than the newer version.
- Supporting dependencies are not fully locked.
- Original checkpoints are not distributed.
- Results concern bounded pixel perturbations on CIFAR-10.
- No certified robustness or deployment-security guarantee is provided.

## Status and future work

The first experiment is complete: baseline training, attacks,
adversarial fine-tuning, independent evaluation, and a clean-only
control comparison.

Future work includes repeated training across seeds, a final-epoch
comparison, and evaluation of additional architectures.