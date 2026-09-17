#!/usr/bin/env bash
exec python mtk_llama2.py \
    --seed 27 \
    --k 10 \
    --n-estimators 500 \
    --max-samples 512 \
    --device cuda:0 \
    "$@"
