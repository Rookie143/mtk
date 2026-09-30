"""Build the Llama2 reference-bank JSONL with the original sampling recipe."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BENIGN_SOURCES = [
    ("llm/datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 300),
    ("llm/datasets/train_data/normal_ood.txt", 300),
    ("llm/datasets/train_data/normal.txt", 300),
    ("llm/datasets/train_data/databricks-dolly-15k.txt", 300),
]

MALICIOUS_SOURCES = [
    ("llm/datasets/train_data/AdvBench.txt", 200),
    ("llm/datasets/train_data/MaliciousInstruct.txt", 200),
    ("llm/datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 200),
    ("llm/datasets/train_data/PKU-SafeRLHF-prompts.txt", 200),
    ("llm/datasets/train_data/UltraSafety.txt", 200),
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
        default=None,
        help="Optional sampling seed. Omit it to match the original unseeded behavior.",
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


def sample_exact(path: str, count: int) -> list[str]:
    lines = read_nonempty(path)
    if len(lines) < count:
        raise ValueError(f"{path} has only {len(lines)} non-empty lines; need {count}")
    return random.sample(lines, count)


def sample_up_to(path: str, count: int) -> list[str]:
    lines = read_nonempty(path)
    return random.sample(lines, min(count, len(lines)))


def write_jsonl(output: Path) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    benign_count = 0
    malicious_count = 0
    with output.open("w", encoding="utf-8") as file:
        for path, count in BENIGN_SOURCES:
            for text in sample_exact(path, count):
                file.write(json.dumps({"text": text, "label": 1}, ensure_ascii=False) + "\n")
                benign_count += 1

        for path, count in MALICIOUS_SOURCES:
            for text in sample_up_to(path, count):
                file.write(json.dumps({"text": text, "label": 0}, ensure_ascii=False) + "\n")
                malicious_count += 1
    return benign_count, malicious_count


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    output = Path(args.output)
    benign_count, malicious_count = write_jsonl(output)
    print("saved_jsonl:", output)
    print("benign_count:", benign_count)
    print("malicious_count:", malicious_count)
    print("total_count:", benign_count + malicious_count)
    print("reference_seed:", args.seed if args.seed is not None else "unseeded")


if __name__ == "__main__":
    main()
