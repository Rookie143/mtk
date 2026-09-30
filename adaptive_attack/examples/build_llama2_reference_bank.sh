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

Default reference-bank composition:
  benign:
    300 Databricks-Dolly samples
    300 Alpaca samples
    600 non-refusal samples
  malicious:
    100 AdvBench samples
    100 MaliciousInstruct samples
    600 PKU-SafeRLHF samples

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

JSONL_PATH="$JSONL_PATH" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["JSONL_PATH"])
output.parent.mkdir(parents=True, exist_ok=True)

benign_sources = [
    ("llm/datasets/train_data/databricks-dolly-15k.txt", 300),
    ("llm/datasets/train_data/alpaca.txt", 300),
    ("llm/datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 600),
]

malicious_sources = [
    ("llm/datasets/train_data/AdvBench.txt", 100),
    ("llm/datasets/train_data/MaliciousInstruct.txt", 100),
    ("llm/datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600),
]


def read_first_nonempty(path: str, count: int) -> list[str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing source file: {source}")
    lines = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < count:
        raise ValueError(f"{source} has only {len(lines)} non-empty lines; need {count}")
    return lines[:count]


benign_count = 0
malicious_count = 0
with output.open("w", encoding="utf-8") as file:
    for path, count in benign_sources:
        for text in read_first_nonempty(path, count):
            file.write(json.dumps({"text": text, "label": 1}, ensure_ascii=False) + "\n")
            benign_count += 1

    for path, count in malicious_sources:
        for text in read_first_nonempty(path, count):
            file.write(json.dumps({"text": text, "label": 0}, ensure_ascii=False) + "\n")
            malicious_count += 1

print("saved_jsonl:", output)
print("benign_count:", benign_count)
print("malicious_count:", malicious_count)
print("total_count:", benign_count + malicious_count)
PY

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
