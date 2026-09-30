#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash adaptive_attack/examples/build_llama2_reference_bank.sh MODEL_PATH [OUTPUT_PT] [JSONL_PATH]

Build the Llama2 reference bank used by the adaptive attack.

Arguments:
  MODEL_PATH   Local Llama2 chat model directory or Hugging Face model id.
  OUTPUT_PT    Output .pt path. Default: adaptive_attack/reference_bank.pt
  JSONL_PATH   Intermediate JSONL path. Default: adaptive_attack/data/reference_bank_llama2.jsonl

Environment variables:
  PYTHON_BIN   Python executable. Default: python
  DEVICE       Device passed to build_reference_bank.py. Default: cuda
  DTYPE        Torch dtype. Default: float16
  BATCH_SIZE   Feature extraction batch size. Default: 4
  JSON_ONLY    Set to 1 to generate only the intermediate JSONL.
  REFERENCE_SEED
              Optional random seed for sampling. Unset matches the original
              unseeded random.sample behavior.

Default reference-bank composition:
  benign:
    300 non-refusal samples
    300 normal_ood samples
    300 normal samples
    300 Databricks-Dolly samples
  malicious:
    up to 200 AdvBench samples
    up to 200 MaliciousInstruct samples
    up to 200 PKU-SafeRLHF 3-6k samples
    up to 200 PKU-SafeRLHF samples
    up to 200 UltraSafety samples

Example:
  CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/python \
    bash adaptive_attack/examples/build_llama2_reference_bank.sh \
      /path/to/Llama-2-7b-chat-hf
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 1 ]]; then
  usage
  exit 2
fi

MODEL_PATH="$1"
OUTPUT_PT="${2:-adaptive_attack/reference_bank.pt}"
JSONL_PATH="${3:-adaptive_attack/data/reference_bank_llama2.jsonl}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "$REPO_ROOT"

echo "[build_llama2_reference_bank] repo: $REPO_ROOT"
echo "[build_llama2_reference_bank] model: $MODEL_PATH"
echo "[build_llama2_reference_bank] jsonl: $JSONL_PATH"
echo "[build_llama2_reference_bank] output: $OUTPUT_PT"
echo "[build_llama2_reference_bank] python: $PYTHON_BIN"
echo "[build_llama2_reference_bank] device: $DEVICE"
echo "[build_llama2_reference_bank] dtype: $DTYPE"
echo "[build_llama2_reference_bank] batch_size: $BATCH_SIZE"

if [[ -n "${REFERENCE_SEED:-}" ]]; then
  "$PYTHON_BIN" -m adaptive_attack.examples.build_llama2_reference_jsonl \
    --output "$JSONL_PATH" \
    --seed "$REFERENCE_SEED"
else
  "$PYTHON_BIN" -m adaptive_attack.examples.build_llama2_reference_jsonl \
    --output "$JSONL_PATH"
fi

if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  echo "[build_llama2_reference_bank] JSON_ONLY=1, skip .pt feature extraction."
  exit 0
fi

"$PYTHON_BIN" -m adaptive_attack.examples.build_reference_bank \
  --model "$MODEL_PATH" \
  --input "$JSONL_PATH" \
  --output "$OUTPUT_PT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE"

echo "[build_llama2_reference_bank] done: $OUTPUT_PT"
