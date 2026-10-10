"""Build the Llama2 reference-bank JSONL with the main MTK recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


BENIGN_SOURCES = [
    ("llm/datasets/train_data/databricks-dolly-15k.txt", 300),
    ("llm/datasets/train_data/alpaca.txt", 300),
    ("llm/datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200),
]

MALICIOUS_SOURCES = [
    ("llm/datasets/train_data/AdvBench.txt", 100),
    ("llm/datasets/train_data/MaliciousInstruct.txt", 100),
    ("llm/datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Llama2 reference-bank JSONL",
    )
    parser.add_argument(
        "--output",
        default="adaptive_attack/data/reference_bank_llama2.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Sampling seed. Default: 27, matching the main Llama2 MTK setting.",
    )
    return parser.parse_args()


def read_nonempty(path: str) -> list[str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing source file: {source}")
    return [
        line
        for line in source.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines(keepends=True)
        if line.strip()
    ]


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_exact(path: str, count: int, seed: int, role: str) -> list[str]:
    lines = read_nonempty(path)
    if len(lines) < count:
        raise ValueError(f"{path} has only {len(lines)} non-empty lines; need {count}")
    rng = random.Random(stable_seed(seed, f"train:{role}:{Path(path).name}"))
    return [lines[index] for index in rng.sample(range(len(lines)), count)]


def write_jsonl(output: Path, seed: int) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    benign_count = 0
    malicious_count = 0
    with output.open("w", encoding="utf-8") as file:
        for path, count in BENIGN_SOURCES:
            for text in sample_exact(path, count, seed, "benign"):
                file.write(json.dumps({"text": text, "label": 1}, ensure_ascii=False) + "\n")
                benign_count += 1

        for path, count in MALICIOUS_SOURCES:
            for text in sample_exact(path, count, seed, "malicious"):
                file.write(json.dumps({"text": text, "label": 0}, ensure_ascii=False) + "\n")
                malicious_count += 1
    return benign_count, malicious_count


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    benign_count, malicious_count = write_jsonl(output, args.seed)
    print("saved_jsonl:", output)
    print("benign_count:", benign_count)
    print("malicious_count:", malicious_count)
    print("total_count:", benign_count + malicious_count)
    print("reference_seed:", args.seed)


if __name__ == "__main__":
    main()
