"""Download the standard AdvBench harmful_behaviors.csv used by GCG."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen


ADV_BENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download standard AdvBench CSV")
    parser.add_argument(
        "--output",
        default="adaptive_attack/data/harmful_behaviors.csv",
        help="Destination CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(ADV_BENCH_URL) as response:
        payload = response.read()

    text = payload.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None or not {"goal", "target"}.issubset(reader.fieldnames):
        raise ValueError("downloaded AdvBench CSV does not contain goal and target columns")

    output.write_text(text, encoding="utf-8")
    print("saved:", output)


if __name__ == "__main__":
    main()
