# Adversarial Robustness Lab

A CIFAR-10 experiment comparing a baseline image classifier,
clean-only fine-tuning, and adversarial fine-tuning.

## Main finding

Adversarial fine-tuning improved accuracy under the standard
AutoAttack suite from 0.06% to 6.72% at an 8/255 pixel-change limit,
while clean accuracy decreased from 70.29% to 63.07%.

A clean-only fine-tuning control improved clean accuracy to 72.05%,
but its AutoAttack accuracy was 0.03%.

This supports the conclusion that adversarial training, rather than
additional ordinary training alone, produced the observed resistance
gain in this experiment.

These are results from one experiment, not security guarantees.

## Three-model test comparison

| Model | Clean accuracy | AutoAttack accuracy at 8/255 | Images surviving AutoAttack |
|---|---:|---:|---:|
| Original baseline | 70.29% | 0.06% | 6 |
| Clean-only fine-tuning control | 72.05% | 0.03% | 3 |
| Adversarial fine-tuning | 63.07% | 6.72% | 672 |

All results use the full 10,000-image test split.

Compared with the baseline, adversarial fine-tuning increased
AutoAttack accuracy by 6.66 percentage points and reduced clean
accuracy by 7.22 percentage points.

The control's three survivors versus the baseline's six should not
be interpreted as an established difference across repeated runs.

## Model and data

- Dataset: CIFAR-10.
- Architecture: a small convolutional neural network.
- Trainable parameters: 545,098.
- Training images: 45,000.
- Validation images: 5,000.
- Test images: 10,000.
- Input dimensions: 3 color channels, 32 pixels high, 32 pixels wide.
- Pixel values: [0, 1].
- Train/validation split seed: 42.

Training images update model parameters.
Validation images select checkpoints.
The separate test split measures the selected models.

## Experiment design

The baseline is trained for 10 epochs, with its checkpoint selected
by highest clean validation accuracy.

Both fine-tuning experiments start from the same baseline checkpoint
and complete five additional epochs.

They share:

- Model architecture.
- Training and validation split.
- Batch size: 128.
- Optimizer: Adam, initialized separately for each experiment.
- Learning rate: 0.0001.
- Training-data shuffle generator seed: 42.
- Validation procedure.
- Checkpoint-selection rule.

For both fine-tuning runs, selection prioritizes PGD validation
survival accuracy, breaking ties using clean validation accuracy.
The starting checkpoint is also a candidate.

The selected clean-only checkpoint is from fine-tuning epoch 1.
The selected adversarial checkpoint is from fine-tuning epoch 5.

Thus, the training budget and selection policy are matched, but the
selected checkpoints have different numbers of fine-tuning epochs.

### Clean-only control

The model learns only from original images:

```text
loss = clean cross-entropy
```

Attacked validation images are used for checkpoint selection.
They do not update model weights.

### Adversarial fine-tuning

PGD generates attacked images using the current model for each
training batch.

The model learns from both original and attacked images:

```text
loss = 0.5 × clean cross-entropy
     + 0.5 × attacked cross-entropy
```

Training attack settings:

- Epsilon: 8/255.
- Step size: 2/255.
- PGD steps: 7.
- One random starting point per image.
- No additional restarts.

Both fine-tuning runs use 20-step PGD for validation.

Adversarial fine-tuning requires more computation per update.
The control matches epochs and optimizer updates, not compute time.

## Threat model

During evaluation attacks, model weights remain fixed.
Attacks modify image pixels within a specified limit.

The limit is the maximum absolute change to any pixel channel,
also called the Linf constraint.

At 8/255, each pixel channel may change by at most approximately
0.031373 on the [0, 1] input scale.

FGSM and our custom PGD are untargeted, white-box attacks using
the correct labels.

The standard AutoAttack suite includes gradient-based and
query-based methods, including targeted components.

## Attack configurations and results

### Fast Gradient Sign Method

- One gradient-based step.
- Cross-entropy loss.
- Resulting pixel values clipped to [0, 1].

Baseline results:

| Maximum pixel-channel change | Test accuracy |
|---|---:|
| No attack | 70.29% |
| 1/255 | 50.18% |
| 2/255 | 33.98% |
| 4/255 | 13.27% |
| 8/255 | 1.54% |

These measurements supersede the earlier baseline results.
FGSM has not been evaluated on the fine-tuned models in this report.

### Projected Gradient Descent

Test configuration:

- Epsilon: 8/255.
- Step size: 2/255.
- Steps: 20.
- Cross-entropy loss.
- One random starting point per image.
- No additional restarts.
- Seed: 42.
- Batch size: 128.

| Model | Clean accuracy | Accuracy after PGD |
|---|---:|---:|
| Baseline | 70.29% | 0.07% |
| Adversarial fine-tuning | 63.07% | 11.52% |

The clean-only control has not received this separate full-test
PGD evaluation.

The defense report also measures survival accuracy: the fraction
of all test images classified correctly both before and after
the attack. It was also 11.52% for the defense.

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

AutoAttack found failures missed by our 20-step PGD evaluation.

Attack success rates among initially correct images:

| Model | Attack success rate |
|---|---:|
| Baseline | 99.91% |
| Clean-only control | 99.96% |
| Adversarial fine-tuning | 89.35% |

Surviving these attacks does not prove that an image is robust.

## Example attack

An initially correct baseline prediction changed from cat to dog
under FGSM at 8/255.

Left: original image.
Middle: attacked image.
Right: magnified absolute pixel differences.

![FGSM example](reports/fgsm_example.png)

This example was selected because the attack succeeded.
The numerical comparisons use the full test split.

## Run locally

Recorded environment:

- Windows.
- Python 3.11.9.
- PyTorch 2.11.0+cu128.
- Torchvision 0.26.0+cu128.
- NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB graphics memory.

Git must be installed to install AutoAttack from its pinned source.
The requirements target a compatible NVIDIA environment.

From the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements pin PyTorch, Torchvision, and the exact AutoAttack
source revision used. Supporting dependencies are not fully locked.

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

The control imports the shared validation function from
`train_adversarial.py`. Importing it does not run defense training.

The control script checks that the starting baseline fingerprint
matches the original experiment. A separately retrained baseline
may have a different fingerprint and require an explicitly updated
experiment configuration before running the control.

The dataset downloads during the first inspection run.
Training creates checkpoints, which are excluded from Git.

Both fine-tuning scripts stop if their output checkpoint or training
history already exists, preventing accidental replacement.

If a checkpoint changes, rerun all evaluations that use it.
Exact training results may vary across environments.

## Checkpoint identities

Reports record SHA256 fingerprints identifying evaluated files.

Baseline:

```text
804dcdd45f32abade22306d8746c229ab3185b107cb7fb8086a10eb5c163801f
```

Adversarially fine-tuned model:

```text
6c9503043e6f9d8b5108277ba92684e4f131e83c92d9d9a129e3f96d43d4fdfc
```

The clean-only checkpoint fingerprint is recorded in
`reports/clean_control_autoattack.json`.

Fingerprints identify file contents. They do not guarantee
deterministic training or model security.

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
| `evaluate_autoattack.py` | Evaluate baseline with AutoAttack |
| `train_adversarial.py` | Train the defense and define shared validation |
| `evaluate_defense.py` | Evaluate defense clean and PGD performance |
| `evaluate_defense_autoattack.py` | Evaluate defense with AutoAttack |
| `train_clean_control.py` | Train the clean-only control |
| `evaluate_clean_control.py` | Evaluate control with AutoAttack |

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
| `reports/clean_control_training.json` | Control configuration and validation history |
| `reports/clean_control_autoattack.json` | Control clean and AutoAttack results |
| `reports/clean_control_autoattack.log` | Control AutoAttack log |

The preliminary trial is excluded from the main comparison.

## Limitations

- One architecture and one run per fine-tuning method.
- Training has not been repeated across multiple seeds.
- Fine-tuning budgets and selection policies match, but selected
  checkpoints have different fine-tuning durations.
- The control does not match adversarial training's computation cost.
- Custom PGD uses one random start and its final iterate rather
  than retaining successful intermediate attacks.
- Supporting dependencies are not fully locked.
- Original checkpoints are not distributed.
- Results concern bounded pixel perturbations on CIFAR-10.
- No certified robustness or deployment-security guarantee is provided.

## Next experiments

- Strengthen custom PGD evaluation with multiple restarts and
  retention of successful intermediate attacks.
- Repeat training across multiple seeds.
- Add an explicitly reported final-epoch comparison.