# MTK Adaptive Attack

This package runs MTK-aware adaptive GCG attacks. It optimizes an adversarial
suffix with two goals:

```text
make the target model generate the target response
make the prompt hidden states look closer to benign reference features
```

The package supports three evasion losses: `l1`, `l2`, and `l3`.

## Install

From the repository root:

```bash
pip install -e ./adaptive_attack
```

For development:

```bash
pip install -e "./adaptive_attack[dev]"
```

## Required Files

You need three local artifacts:

| File | Purpose |
| --- | --- |
| target model | Hugging Face causal LM directory or model ID |
| `reference_bank.pt` | MTK hidden-state reference bank |
| sample JSON | attack prompts and target response prefixes |

Large model weights, `.pt` feature banks, and generated result files should not be
committed to Git.

## Reference Bank Format

`reference_bank.pt` is a PyTorch dictionary:

```python
{
    "background_layered_activations": features,
    "labels": labels,
}
```

Expected shapes:

```text
features: Tensor[num_samples, num_layers, hidden_size]
labels:   Tensor[num_samples]
```

Default label convention:

```text
1 = benign
0 = malicious
```

The reference bank must be extracted with the same model, tokenizer, chat
template, layer count, and hidden-state position used for the attack.

To build a bank from a JSONL file:

```bash
python -m adaptive_attack.examples.build_reference_bank \
  --model /path/to/model \
  --input /path/to/reference.jsonl \
  --output /path/to/reference_bank.pt \
  --device cuda \
  --dtype float16 \
  --batch-size 4
```

Input JSONL format:

```json
{"text": "benign reference text", "label": 1}
{"text": "malicious reference text", "label": 0}
```

## Attack Sample Format

Batch scripts expect a JSON list. Each item needs at least:

```json
[
  {
    "goal": "user prompt to optimize against",
    "target": "target response prefix"
  }
]
```

Optional fields such as `method` and `model` are preserved in `raw_results.jsonl`.
A placeholder example is provided at:

```text
examples/sample_advbench_format.json
```

## Run One Attack

Use `run_mtk.py` for a single prompt:

```bash
python -m adaptive_attack.examples.run_mtk \
  --model /path/to/model \
  --feature-library /path/to/reference_bank.pt \
  --prompt "YOUR_PROMPT" \
  --target "YOUR_TARGET_RESPONSE_PREFIX" \
  --loss-type l3 \
  --lambda 0.1 \
  --num-steps 500 \
  --search-width 128 \
  --topk 128
```

## Run A Batch Sweep

Use the batch script to run multiple samples, losses, and lambda values:

```bash
python -m adaptive_attack.examples.run_llama2_batch \
  --model /path/to/Llama-2-7b-chat-hf \
  --feature-library /path/to/reference_bank.pt \
  --sample-file /path/to/samples.json \
  --output-dir ./adaptive_attack_results \
  --max-samples 30 \
  --loss-types l3 \
  --lambdas 0.1,0.3,0.5,0.7,0.9 \
  --num-steps 500 \
  --search-width 128 \
  --topk 128 \
  --batch-size 128
```

The script resumes by default. Completed `(loss_type, lambda, sample_index)` rows
in `raw_results.jsonl` are skipped. Use `--no-resume` to force a fresh run.

## Common Options

| Option | Meaning |
| --- | --- |
| `--loss-types` | Comma-separated loss names: `l1`, `l2`, `l3` |
| `--lambdas` | Comma-separated evasion weights |
| `--num-steps` | Number of GCG optimization steps per sample |
| `--search-width` | Number of candidate suffixes evaluated per step |
| `--topk` | Token-gradient candidate pool size |
| `--batch-size` | Number of candidates evaluated in one forward pass |
| `--detector-max-anchors` | Optional balanced anchor subset for faster detector scoring |
| `--no-detector` | Skip MTK detector scoring and report ASR only |
| `--exclude-generation` | Do not save generated model responses |
| `--success-judge` | `loose` by default, or `prefix` for strict target-prefix matching |

Environment-variable defaults are also supported:

| Environment variable | Used as |
| --- | --- |
| `MTK_ADAPTIVE_MODEL` | default `--model` |
| `MTK_ADAPTIVE_BANK` | default `--feature-library` |
| `MTK_ADAPTIVE_SAMPLE_FILE` | default `--sample-file` |
| `MTK_ADAPTIVE_OUTPUT_DIR` | default batch `--output-dir` |
| `MTK_ADAPTIVE_SINGLE_OUTPUT` | default single-run `--output` |

## Loss Types

The adaptive objective is:

```text
L = (1 - lambda) * sequence_loss + lambda * evasion_loss
```

| Loss | Behavior |
| --- | --- |
| `l1` | Pull hidden states toward all benign anchors |
| `l2` | Pull hidden states toward the nearest benign anchor |
| `l3` | Pull hidden states toward benign anchors and away from malicious anchors |

## Outputs

The batch script writes:

| File | Contents |
| --- | --- |
| `raw_results.jsonl` | One JSON record per attack |
| `summary.csv` | Machine-readable aggregate metrics |
| `summary.md` | Human-readable aggregate table |

Each successful raw row contains fields such as:

```text
best_suffix
generation
best_loss
sequence_loss
feature_loss
detector_score
detector_prediction
attack_success
attack_success_prefix
attack_success_loose
detected_by_mtk
effective_attack_success
```

`generation` is saved by default so runs can be inspected. Add
`--exclude-generation` if you do not want to retain model outputs.

## Metrics

| Metric | Meaning |
| --- | --- |
| `ASR` | Attack success under the selected `--success-judge` |
| `ASR_prefix` | Strict success: generation starts with the exact target prefix |
| `ASR_loose` | Looser success: affirmative, target-like generation |
| `TPR` | Fraction of attacks detected by MTK |
| `eASR` | Attack succeeds and is not detected |
| `eASR_prefix` | Effective ASR under the strict prefix judge |
| `eASR_loose` | Effective ASR under the loose judge |

For quick ASR-only speed tests, add:

```bash
--no-detector
```

Then `TPR` and `eASR` are reported as `NA`.

## Regenerate Summaries

If `raw_results.jsonl` already exists, regenerate only the summary files with:

```bash
python -m adaptive_attack.examples.run_llama2_batch \
  --output-dir ./adaptive_attack_results \
  --summary-only
```

## Tests

```bash
pytest -q adaptive_attack/tests
```
