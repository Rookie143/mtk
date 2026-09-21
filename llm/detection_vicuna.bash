#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$script_dir/mtk_vicuna.py" \
    --seed 56 \
    --colon-k 10 \
    --colon-n-estimators 500 \
    --colon-max-samples 512 \
    --ist-k 10 \
    --ist-n-estimators 500 \
    --ist-max-samples 512 \
    --ist-weight 0.25 \
    --device cuda:0 \
    "$@"
