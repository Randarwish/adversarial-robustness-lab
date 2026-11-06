# Adversarial Robustness Lab

A reproducible experiment comparing an image classifier before and
after adversarial fine-tuning on CIFAR-10.

## Main finding

Five epochs of adversarial fine-tuning increased test accuracy
under the standard AutoAttack suite from 0.06% to 6.72% at an
8/255 pixel-change limit.

Clean test accuracy decreased from 70.29% to 63.07%.

This is a measured improvement in attack resistance, with substantial
vulnerability remaining. It is not a security guarantee.

## Model and dataset

- Dataset: CIFAR-10.
- Model: a small convolutional neural network.
- Trainable parameters: 545,098.
- Training split: 45,000 images.
- Validation split: 5,000 images.
- Test split: 10,000 images.
- Input pixel values: [0, 1].
- Baseline checkpoint selection: highest clean validation accuracy.
- Defense checkpoint selection: highest PGD validation survival
  accuracy, breaking ties using clean validation accuracy.

Training and validation use a fixed split with seed 42.
The separate test split is used for final evaluation.

## Baseline versus defense

| Test evaluation | Baseline | Adversarially fine-tuned |
|---|---:|---:|
| Clean accuracy | 70.29% | 63.07% |
| PGD accuracy at 8/255 | 0.07% | 11.52% |
| AutoAttack accuracy at 8/255 | 0.06% | 6.72% |

All evaluations use the full 10,000-image test split.

The defense increased AutoAttack accuracy by 6.66 percentage points
and reduced clean accuracy by 7.22 percentage points.

The defense retained 672 correct predictions after AutoAttack,
compared with 6 for the baseline.

Under AutoAttack, 89.35% of the defense's initially correct
predictions became incorrect.

## Baseline FGSM results

| Maximum pixel-channel change | Test accuracy |
|---|---:|
| No attack | 70.29% |
| 1/255 | 50.18% |
| 2/255 | 33.98% |
| 4/255 | 13.27% |
| 8/255 | 1.54% |

These results supersede the earlier baseline measurements.

## Threat model

During evaluation attacks, the model's weights remain fixed.
The attacker modifies input pixels within a specified limit.

FGSM and the custom PGD implementation are untargeted, white-box
attacks using the correct labels.

The standard AutoAttack suite includes gradient-based and
query-based methods, including targeted components.

All reported adversarial evaluations use the maximum absolute
pixel-channel change, also called the Linf constraint.
Images are constrained to the valid pixel range [0, 1].

## Attack configurations

### Fast Gradient Sign Method

- One gradient-based step.
- Cross-entropy loss.
- Epsilon values: 0, 1/255, 2/255, 4/255, and 8/255.

### Projected Gradient Descent

- Epsilon: 8/255.
- Step size: 2/255.
- Steps: 20.
- Cross-entropy loss.
- One random starting point per image.
- No additional restarts.
- Seed: 42 for test evaluation.
- Batch size: 128.

The defense evaluation reports both accuracy after the attack
and survival accuracy: the fraction of all test images classified
correctly both before and after the attack.

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

Later components attack examples that remain correctly classified
after earlier components.

For the defense, accuracy surviving the suite changed as follows:

| Stage | Survival accuracy |
|---|---:|
| Clean images | 63.07% |
| After APGD-CE | 9.91% |
| After APGD-T | 6.73% |
| After FAB-T | 6.72% |
| After Square Attack | 6.72% |

The stronger suite found failures missed by our 20-step PGD
evaluation.

Surviving these attacks does not prove that an image is robust.

## Adversarial fine-tuning

The defense starts from the trained baseline checkpoint.
It does not train a second model from scratch.

Configuration:

- Epochs: 5.
- Optimizer: Adam.
- Learning rate: 0.0001.
- Batch size: 128.
- Training attack: 7-step PGD.
- Training epsilon: 8/255.
- Training step size: 2/255.
- One random starting point per image.
- Loss: 50% clean cross-entropy and 50% attacked cross-entropy.
- Validation attack: 20-step PGD at the same epsilon.
- Starting checkpoint included as a selection candidate.

Adversarial examples are regenerated using the current model for
each training batch.

Only training images update model parameters.
Validation performance selects the defense checkpoint.

## Example attack

An initially correct baseline prediction changed from cat to dog
under FGSM at 8/255.

From left to right:

1. Original image.
2. Attacked image.
3. Magnified absolute pixel differences.

![FGSM example](reports/fgsm_example.png)

This is a selected successful attack example.
The numerical evaluations cover the full test split.

## Checkpoint identities

The evaluation reports record checkpoint SHA256 fingerprints.

Baseline:

```text
804dcdd45f32abade22306d8746c229ab3185b107cb7fb8086a10eb5c163801f
```

Adversarially fine-tuned model:

```text
6c9503043e6f9d8b5108277ba92684e4f131e83c92d9d9a129e3f96d43d4fdfc
```

Fingerprints identify the evaluated files. They do not make
training deterministic or certify model security.

## Run locally

Recorded environment:

- Windows.
- Python 3.11.9.
- PyTorch 2.11.0+cu128.
- Torchvision 0.26.0+cu128.
- NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB graphics memory.

The following installation uses the CUDA 12.8 package index
for a compatible NVIDIA environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install git+https://github.com/fra31/auto-attack
```

The AutoAttack installation command uses its current default
revision. The exact source revision remains to be documented.

### Train and evaluate the baseline

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

### Train and evaluate the defense

```powershell
.\.venv\Scripts\python.exe train_adversarial.py
.\.venv\Scripts\python.exe evaluate_defense.py
.\.venv\Scripts\python.exe evaluate_defense_autoattack.py
```

The defense training script stops if its checkpoint or training
history already exists, preventing accidental replacement.

The dataset downloads during the first inspection run.
Checkpoints are created by training and are excluded from Git.

Retraining can produce different results.
If a checkpoint changes, rerun all evaluations that use it.

## Project files

| File | Purpose |
|---|---|
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
| `train_adversarial.py` | Fine-tune using clean and attacked images |
| `evaluate_defense.py` | Evaluate defense clean and PGD performance |
| `evaluate_defense_autoattack.py` | Evaluate defense with AutoAttack |

## Reports

| File | Contents |
|---|---|
| `reports/baseline.json` | Clean baseline test results |
| `reports/fgsm.json` | Baseline FGSM results |
| `reports/pgd.json` | Baseline PGD results |
| `reports/fgsm_example.png` | Selected baseline attack example |
| `reports/autoattack_trial.json` | Preliminary 100-image evaluation |
| `reports/autoattack_trial.log` | Preliminary evaluation log |
| `reports/autoattack_full.json` | Full baseline AutoAttack results |
| `reports/autoattack_full.log` | Full baseline AutoAttack log |
| `reports/adversarial_training.json` | Defense configuration and validation history |
| `reports/defense_pgd.json` | Defense clean and PGD test results |
| `reports/defense_autoattack.json` | Defense AutoAttack test results |
| `reports/defense_autoattack.log` | Defense AutoAttack log |

The preliminary trial is separate from the full-test results
and is not used in the main comparison.

## Limitations

- One model architecture and one baseline/defense checkpoint pair.
- No repeated training across multiple seeds.
- Defense uses additional training beyond the baseline; there is
  no matched clean-only fine-tuning control.
- Custom PGD evaluation uses one random start and its final iterate,
  rather than retaining every successful intermediate attack.
- Exact AutoAttack source revision is not yet pinned.
- Original checkpoints are not distributed in this repository.
- Results concern bounded pixel perturbations on CIFAR-10 and do
  not establish real-world deployment security.
- No certified robustness guarantee is provided.

## Next experiments

- Record and pin the installed AutoAttack source revision.
- Add a matched clean-only fine-tuning control.
- Evaluate across multiple training seeds.
- Strengthen custom PGD evaluation with multiple restarts and
  retention of successful intermediate attacks.