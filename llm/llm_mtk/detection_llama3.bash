#!/usr/bin/env bash
exec python mtk_llama3.py \
    --seed 35 \
    --k 10 \
    --n-estimators 500 \
    --max-samples 512 \
    --device cuda:0 \
    "$@"
