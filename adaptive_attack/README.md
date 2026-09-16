# MTK Adaptive Attack

This package provides MTK-aware adaptive GCG attacks with three hidden-state evasion
losses: `L1`, `L2`, and `L3`.

The directory is self-contained. It can be copied, installed, and used without the
original `nanogcg` directory and without running the full MTK detection benchmark.

## Scope

`adaptive_attack` focuses on the adaptive attack workflow:

1. Build a hidden-state reference bank from the target model.
2. Load the reference bank as MTK detector/scorer features.
3. Add `L1`, `L2`, or `L3` hidden-state evasion loss to GCG optimization.
4. Produce an adversarial suffix that optimizes both the target response objective
   and the MTK evasion objective.

This package is not the full MTK detection benchmark. To reproduce MTK detector
AUROC, FPR, TPR, or dataset-level LLM/VLM detection results, use the outer `llm/`
and `vlm/` directories in the full repository.

## Mapping To The Full MTK Repository

| Full MTK repository | This package |
| --- | --- |
| `extract_trainset_hiddenstates_*.py` | `examples/build_reference_bank.py` |
| `JailbreakDetector_*.py` | `MTKDetector` in `mtk.py` |
| MTK hidden-state scoring | `TorchMTKScorer` in `mtk.py` |
| MTK reference features | `HiddenStateLibrary` in `feature_store.py` |
| Adaptive `L1`/`L2`/`L3` losses | `losses.py` and `objectives.py` |
| GCG optimization loop | `gcg_core.py`, `gcg_utils.py`, and `engine.py` |

## Workflow

1. Prepare a JSONL reference dataset with text and labels.
2. Generate `reference_bank.pt` with `examples/build_reference_bank.py`.
3. Run the adaptive attack with `examples/run_mtk.py` and choose `l1`, `l2`, or `l3`.
4. Optionally use `MTKDetector` to score candidate hidden states.

## Installation

Run the following command from the parent directory of `adaptive_attack`:

```bash
pip install -e ./adaptive_attack
```

For development and tests:

```bash
pip install -e "./adaptive_attack[dev]"
```

## Reference Bank

`--feature-library` points to a PyTorch `.pt` file. This file is not model weights.
It is the MTK hidden-state reference bank: a set of benign and malicious reference
samples passed through the target model, with their layer-wise hidden states and
labels saved.

The reference bank is used for two purposes:

1. The MTK detector computes K-NB rank trajectories from the reference samples.
2. The adaptive `L1`, `L2`, and `L3` losses compute hidden-state distances against
   the reference samples.

The package loads and uses the `.pt` file, but it does not download any datasets or
prebuilt reference banks. Build the reference bank with the same model, tokenizer,
chat template, layer count, and hidden-state extraction position as the attacked
model.

The expected file format is a PyTorch dictionary:

```python
{
    "background_layered_activations": features,
    "labels": labels,
}
```

Expected tensor shapes:

```text
features: Tensor[N, L, H]
labels:   Tensor[N]
```

- `N`: number of reference samples
- `L`: number of hidden-state layers
- `H`: model hidden size
- `labels == 1`: benign sample
- `labels == 0`: malicious sample

The reference bank must match the attacked model. A bank generated with a different
model, tokenizer, chat template, layer count, or hidden-state extraction position is
not valid for the attack.

### Build A Reference Bank

The package includes a reference-bank builder. The input is a JSONL file where each
line contains one text sample and one label:

```json
{"text": "A normal user question.", "label": 1}
{"text": "An authorized safety evaluation prompt.", "label": 0}
```

Run:

```bash
python -m adaptive_attack.examples.build_reference_bank \
    --model /path/to/model \
    --input /path/to/reference.jsonl \
    --output /path/to/reference_bank.pt \
    --device cuda \
    --dtype float16 \
    --batch-size 4
```

The script extracts the hidden state of the last input token from every model layer
and saves:

```text
background_layered_activations: [samples, layers, hidden_size]
labels:                         [samples]
```

The default label convention is `1=benign` and `0=malicious`. If your JSONL file uses
different field names, pass `--text-key` and `--label-key`.

Minimal manual example:

```python
import torch

features = torch.stack(reference_hidden_states)
labels = torch.tensor(reference_labels, dtype=torch.long)

torch.save(
    {
        "background_layered_activations": features,
        "labels": labels,
    },
    "reference_bank.pt",
)
```

Each item in `reference_hidden_states` should have shape `[layers, hidden_size]`.
After stacking, `features` should have shape `[samples, layers, hidden_size]`.

When extracting hidden states, use the representation of the last input token from
each target model layer. Do not save language-model logits or generated text as the
reference bank.

## Losses

The adaptive objective is:

```text
L_adapt = (1 - lambda) * L_adv + lambda * L_evasion
```

Select `L_evasion` with `loss_type`:

| Parameter | Meaning |
| --- | --- |
| `l1` | Pull the candidate toward the mean MSE of all benign anchors. |
| `l2` | Pull the candidate toward the nearest benign anchor. |
| `l3` | Pull the candidate toward benign anchors and push it away from malicious anchors. |

## Python API

```python
from adaptive_attack import (
    AdaptiveGCGConfig,
    load_hidden_state_library,
    run_mtk_attack,
)

library = load_hidden_state_library(
    "/path/to/reference_bank.pt",
    map_location="cuda",
)

lambda_value = 0.1
config = AdaptiveGCGConfig(
    num_steps=1000,
    sequence_weight=1.0 - lambda_value,
    feature_weight=lambda_value,
    search_width=512,
    topk=256,
    seed=42,
)

result = run_mtk_attack(
    model=model,
    tokenizer=tokenizer,
    messages="YOUR_PROMPT",
    target="YOUR_TARGET",
    feature_library=library,
    loss_type="l3",
    config=config,
)

print(result.best_string)
print(result.sequence_loss)
print(result.feature_loss)
```

## Command Line

```bash
python -m adaptive_attack.examples.run_mtk \
    --model /path/to/model \
    --feature-library /path/to/reference_bank.pt \
    --prompt "YOUR_PROMPT" \
    --target "YOUR_TARGET" \
    --loss-type l3 \
    --lambda 0.1 \
    --num-steps 1000 \
    --search-width 512 \
    --topk 256
```

Main options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--loss-type` | `l3` | `l1`, `l2`, or `l3` |
| `--lambda` | unset | Set sequence-loss and MTK-loss weights automatically. |
| `--num-steps` | `1000` | Number of GCG optimization steps. |
| `--search-width` | `512` | Number of candidates per optimization step. |
| `--topk` | `256` | Number of token-gradient candidates. |
| `--device` | `cuda` | Device used to run the model. |
| `--dtype` | `float16` | Model dtype. |

If `--lambda` is not used, set the two loss weights directly with
`--sequence-weight` and `--feature-weight`.

## MTK Detector

```python
from adaptive_attack import MTKDetector, load_hidden_state_library

library = load_hidden_state_library(
    "/path/to/reference_bank.pt",
    map_location="cuda",
)

detector = MTKDetector.from_library(library)
scores = detector.decision_function(candidate_features)
predictions = detector.predict(candidate_features)
```

`candidate_features` must have shape:

```text
Tensor[batch, layers, hidden]
```

`predict` returns `1` for benign and `0` for detector-flagged anomalous samples.

## Tests

After installing development dependencies, run from the parent directory:

```bash
pytest -q adaptive_attack/tests
```

Or run from this directory:

```bash
python -m pytest -q tests
```
