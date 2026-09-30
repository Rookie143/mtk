# MTK Adaptive Attack

This directory provides scripts for running MTK-aware adaptive GCG attacks on
Llama2.

## Quick validation result

The quick validation below uses Llama2, 30 AdvBench samples, the `L3` adaptive
loss, 500 GCG steps, and lambda values `0.1`, `0.3`, `0.5`, `0.7`, and `0.9`.
All reported attack-success metrics use the loose judge.

| Loss | Lambda | Samples | ASR loose | TPR | eASR loose |
| --- | ---: | ---: | ---: | ---: | ---: |
| L3 | 0.1 | 30 | 80.0% (24/30) | 96.7% (29/30) | 3.3% (1/30) |
| L3 | 0.3 | 30 | 70.0% (21/30) | 96.7% (29/30) | 3.3% (1/30) |
| L3 | 0.5 | 30 | 60.0% (18/30) | 100.0% (30/30) | 0.0% (0/30) |
| L3 | 0.7 | 30 | 60.0% (18/30) | 96.7% (29/30) | 3.3% (1/30) |
| L3 | 0.9 | 30 | 56.7% (17/30) | 100.0% (30/30) | 0.0% (0/30) |

`ASR loose` is the jailbreak success rate before MTK filtering. `TPR` is the MTK
detection rate on adversarial prompts. `eASR loose` is the jailbreak success
rate after MTK filtering.

## Reproduce the quick validation

Run from the repository root.

```bash
pip install -e ./adaptive_attack
```

Set the local Llama2 model path:

```bash
export LLAMA2_MODEL=/path/to/Llama-2-7b-chat-hf
```

Build the MTK reference bank:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash adaptive_attack/examples/build_llama2_reference_bank.sh \
  "$LLAMA2_MODEL" \
  adaptive_attack/reference_bank.pt
```

Run the 30-sample quick validation:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m adaptive_attack.examples.run_llama2_batch \
  --model "$LLAMA2_MODEL" \
  --feature-library adaptive_attack/reference_bank.pt \
  --sample-file llm/datasets/llama2_test/nanogcg_1.json \
  --output-dir adaptive_attack/repro_llama2_l3_trend_30s500 \
  --max-samples 30 \
  --loss-types l3 \
  --lambdas 0.1,0.3,0.5,0.7,0.9 \
  --num-steps 500 \
  --search-width 128 \
  --topk 128 \
  --batch-size 128 \
  --success-judge loose
```

The main outputs are:

```text
adaptive_attack/repro_llama2_l3_trend_30s500/raw_results.jsonl
adaptive_attack/repro_llama2_l3_trend_30s500/summary.csv
adaptive_attack/repro_llama2_l3_trend_30s500/summary.md
```

## Larger AdvBench validation

Download the standard AdvBench CSV:

```bash
python -m adaptive_attack.examples.download_advbench \
  --output adaptive_attack/data/harmful_behaviors.csv
```

Run a larger sweep. Change `--max-samples` to the desired sample count, or remove
it to evaluate the full CSV.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m adaptive_attack.examples.run_llama2_batch \
  --model "$LLAMA2_MODEL" \
  --feature-library adaptive_attack/reference_bank.pt \
  --sample-file adaptive_attack/data/harmful_behaviors.csv \
  --output-dir adaptive_attack/repro_llama2_advbench_sweep \
  --max-samples 100 \
  --loss-types l1,l2,l3 \
  --lambdas 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --num-steps 1000 \
  --search-width 512 \
  --topk 256 \
  --batch-size 128 \
  --success-judge loose
```
