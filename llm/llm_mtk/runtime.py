import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
LLM_DIR = REPO_DIR.parent


def parse_args(model, seed, k, n_estimators, max_samples):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--k", type=int, default=k)
    parser.add_argument("--n-estimators", type=int, default=n_estimators)
    parser.add_argument("--max-samples", type=int, default=max_samples)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(Path("table4_seed_reproducible") / "runs" / model))
    args = parser.parse_args()
    if args.k < 1 or args.n_estimators < 1 or args.max_samples < 1:
        parser.error("k, n-estimators and max-samples must be positive")
    return args


def configure(args):
    os.chdir(REPO_DIR)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def label_from_path(file_path):
    suffix = Path(file_path).stem.rsplit("_", 1)[-1]
    if suffix not in {"0", "1"}:
        raise ValueError(f"Dataset filename must end in _0.json or _1.json: {file_path}")
    return int(suffix)
