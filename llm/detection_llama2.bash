#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$script_dir/reproduce_best_auroc.py" llama2 \
    --seed 27 \
    --k 10 \
    --n-estimators 500 \
    --max-samples 512 \
    --device cuda:0 \
    "$@"
