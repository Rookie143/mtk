from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from feature_protocol import FeatureProtocol, rank_features, read_lines, set_determinism
from JailbreakDetector_llama2 import JailbreakDetector
from llama2_memory_extract import extract


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "llm"
MODEL = PROJECT / "model" / "llama2"
TRAIN_DATA = PROJECT / "datasets" / "train_data"
PMP_DATA = HERE / "dataset" / "pmp_samples.txt"
GCG_DATA = HERE / "dataset" / "gcg_samples.json"
OUTPUT = HERE / "results" / "results.csv"
DEVICE = "cuda:0"
DTYPE = "float16"
BATCH_SIZE = 16
RANK_BATCH_SIZE = 64
CPU_THREADS = 4
K = 10
N_ESTIMATORS = 500
MAX_SAMPLES = 512
SEED = 21
FEATURE_SEED = 42


def derived_seed(namespace):
    digest = hashlib.sha256(f"{SEED}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def prompt_key(prompt):
    return " ".join(prompt.split())


def rows_from_text(path):
    return [{"prompt": value} for value in read_lines(path)]


def rows_from_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("samples"), list):
        value = value["samples"]
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {path}")
    rows = []
    for item in value:
        if isinstance(item, str):
            prompt = item
            row = {"prompt": prompt}
        elif isinstance(item, dict):
            prompt = item.get("jailbreak") or item.get("prompt")
            row = {"prompt": prompt, "goal": item.get("goal", "")}
        else:
            continue
        if isinstance(prompt, str) and prompt.strip():
            rows.append(row)
    if not rows:
        raise ValueError(f"No usable prompts in {path}")
    return rows


def select_rows(rows, count, namespace, used):
    generator = random.Random(derived_seed(namespace))
    initial = generator.sample(range(len(rows)), min(count, len(rows)))
    initial_set = set(initial)
    remaining = [index for index in range(len(rows)) if index not in initial_set]
    order = initial + generator.sample(remaining, len(remaining))
    selected = []
    for index in order:
        row = rows[index]
        identity = prompt_key(row["prompt"])
        if identity in used:
            continue
        used.add(identity)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"{namespace} contains only {len(selected)} usable unique prompts")
    return selected


def select_test_rows(rows, namespace, used):
    available = []
    seen = set(used)
    for row in rows:
        identity = prompt_key(row["prompt"])
        if identity not in seen:
            seen.add(identity)
            available.append(row)
    return select_rows(rows, min(len(available), 500), namespace, used)


def build_groups():
    dolly = rows_from_text(TRAIN_DATA / "databricks-dolly-15k.txt")
    alpaca = rows_from_text(TRAIN_DATA / "alpaca.txt")
    pmp_source = rows_from_text(TRAIN_DATA / "non_refusal_prompts_with_responses_80k.txt")
    pmp_test_source = rows_from_text(PMP_DATA)
    gcg_source = rows_from_json(GCG_DATA)
    used = set()
    benign = []
    for rows, count, name in (
        (dolly, 300, "databricks-dolly-15k.txt"),
        (alpaca, 300, "alpaca.txt"),
        (pmp_source, 200, "non_refusal_prompts_with_responses_80k.txt"),
    ):
        benign.extend(select_rows(rows, count, f"train:benign:{name}", used))
    malicious = []
    for name, count in (
        ("AdvBench.txt", 100),
        ("MaliciousInstruct.txt", 100),
        ("PKU-SafeRLHF-prompts_3-6k.txt", 600),
    ):
        malicious.extend(
            select_rows(rows_from_text(TRAIN_DATA / name), count, f"train:malicious:{name}", used)
        )
    calibration = select_rows(dolly, 200, "table3:calibration:dolly", used)
    ordinary = select_rows(dolly, 500, "table3:test:dolly", used)
    pmp = select_test_rows(pmp_test_source, "table3:test:pmp", used)
    attacks = select_test_rows(gcg_source, "test:gcg_samples", used)
    return {
        "benign_anchor": benign,
        "malicious_anchor": malicious,
        "calibration": calibration,
        "ordinary": ordinary,
        "pmp": pmp,
        "attack": attacks,
    }


def evaluate(positive, negative, threshold):
    tp = int((positive > threshold).sum())
    fp = int((negative > threshold).sum())
    return {
        "tpr": tp / len(positive),
        "fpr": fp / len(negative),
    }


def run():
    torch.set_num_threads(CPU_THREADS)
    set_determinism(FEATURE_SEED)
    groups = build_groups()
    ordered_roles = (
        "benign_anchor",
        "malicious_anchor",
        "calibration",
        "ordinary",
        "pmp",
        "attack",
    )
    prompts = [row["prompt"] for role in ordered_roles for row in groups[role]]
    protocol = FeatureProtocol(MODEL)
    model, tokenizer = protocol.model_and_tokenizer(DEVICE, DTYPE)
    features = extract(model, tokenizer, prompts, BATCH_SIZE, DEVICE)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    train_count = len(groups["benign_anchor"]) + len(groups["malicious_anchor"])
    labels = torch.tensor(
        [0] * len(groups["benign_anchor"]) + [1] * len(groups["malicious_anchor"])
    )
    train_features = features[:train_count]
    test_features = features[train_count:]
    rank_data = {
        "labels": labels,
        "train_ranks": rank_features(
            train_features,
            train_features,
            labels,
            (K,),
            DEVICE,
            exclude_self=True,
            batch_size=RANK_BATCH_SIZE,
        ),
        "test_ranks": rank_features(
            test_features,
            train_features,
            labels,
            (K,),
            DEVICE,
            batch_size=RANK_BATCH_SIZE,
        ),
    }
    detector = JailbreakDetector(
        rank_data,
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        max_samples=MAX_SAMPLES,
        k_nb=K,
    )
    scores = -detector.score_all()
    score_groups = {}
    offset = 0
    for role in ("calibration", "ordinary", "pmp", "attack"):
        count = len(groups[role])
        score_groups[role] = scores[offset : offset + count]
        offset += count
    threshold = float(np.quantile(score_groups["calibration"], 0.95, method="linear"))
    ordinary = evaluate(score_groups["attack"], score_groups["ordinary"], threshold)
    pmp = evaluate(score_groups["attack"], score_groups["pmp"], threshold)
    f1 = 2 * ordinary["tpr"] / (1 + ordinary["tpr"] + ordinary["fpr"])
    result = {
        "tpr": ordinary["tpr"],
        "fpr": ordinary["fpr"],
        "pmp_fpr": pmp["fpr"],
        "f1": f1,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(f"Results written to {OUTPUT}")


if __name__ == "__main__":
    run()
