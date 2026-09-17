#!/usr/bin/env bash
exec python mtk_mistral.py \
    --seed 47 \
    --k 10 \
    --n-estimators 500 \
    --max-samples 512 \
    --device cuda:0 \
    "$@"
