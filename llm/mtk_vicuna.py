from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Indicator_analysis_drawing import generate_report
from JailbreakDetector_vicuna import JailbreakDetector
from draw_auroc import evaluate_attack_auroc
from extract_AC_json import extract_accuracy_to_excel
from extract_trainset_hiddenstates_vicuna import (
    configure_tokenizer,
    extract_input_ids_activations,
    extract_trainset_hiddenstates,
)
from utils.string_utils import autodan_SuffixManager, load_conversation_template


HERE = Path(__file__).resolve().parent
REPO = HERE
IST_WEIGHT = 0.25


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Vicuna dual-endpoint MTK reproduction"
    )
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--colon-k", type=int, default=10)
    parser.add_argument("--colon-n-estimators", type=int, default=500)
    parser.add_argument("--colon-max-samples", type=int, default=512)
    parser.add_argument("--ist-k", type=int, default=10)
    parser.add_argument("--ist-n-estimators", type=int, default=500)
    parser.add_argument("--ist-max-samples", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--colon-batch-size", type=int, default=8)
    parser.add_argument("--ist-batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default="results/vicuna")
    args = parser.parse_args(argv)
    positive = (
        args.colon_k,
        args.colon_n_estimators,
        args.colon_max_samples,
        args.ist_k,
        args.ist_n_estimators,
        args.ist_max_samples,
        args.colon_batch_size,
        args.ist_batch_size,
    )
    if any(value < 1 for value in positive):
        parser.error("k, n-estimators, and max-samples must be positive")
    return args


def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def method_name(file_path):
    stem = Path(file_path).stem
    return stem[:-2] if stem.endswith(("_0", "_1")) else stem


def load_prompts_from_attack_json(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
        data = json.load(stream)
    suffix = Path(file_path).stem.rsplit("_", 1)[-1]
    if suffix not in {"0", "1"}:
        raise ValueError(f"Dataset filename must end in _0 or _1: {file_path}")
    records = []
    for source_index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = item.get("jailbreak") or item.get("prompt")
        if isinstance(prompt, str) and prompt:
            records.append({
                "prompt": prompt,
                "true_label": int(suffix),
                "source_index": source_index,
            })
    return records


def deterministic_sample(rows, count, seed, namespace):
    if len(rows) <= count:
        return rows
    rng = random.Random(stable_seed(seed, namespace))
    return [rows[index] for index in rng.sample(range(len(rows)), count)]


def autodan_input_ids(tokenizer, file_path, records):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
        raw_items = json.load(stream)
    if not isinstance(raw_items, list):
        raise ValueError(f"AutoDAN dataset must be a JSON list: {file_path}")

    conversation = load_conversation_template("vicuna")
    sequences = []
    for record in records:
        item = raw_items[record["source_index"]]
        goal = (item.get("goal") or item.get("instruction") or "").strip()
        jailbreak = (item.get("jailbreak") or "").strip()
        suffix = jailbreak[len(goal):].strip() if len(jailbreak) >= len(goal) else jailbreak
        manager = autodan_SuffixManager(
            tokenizer=tokenizer,
            conv_template=conversation,
            instruction=goal,
            target=item.get("target"),
            adv_string=suffix,
        )
        sequences.append(manager.get_input_ids(adv_string=suffix))
    return sequences


def get_train_dataset(benign_path_list, malicious_path_list, seed):
    selected = []
    for role, path_list in (("benign", benign_path_list), ("malicious", malicious_path_list)):
        role_prompts = []
        for file_path, count in path_list:
            with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as stream:
                rows = [line for line in stream.readlines() if line.strip()]
            rng = random.Random(stable_seed(seed, f"train:{role}:{Path(file_path).name}"))
            role_prompts.extend(rows[index] for index in rng.sample(range(len(rows)), count))
        selected.append(role_prompts)
    return selected


def eval(attack_file_path_list, detector, your_flag, seed):
    for file_path in attack_file_path_list:
        attack_key = f"vicuna_test_{Path(file_path).stem}"
        report_path = Path(your_flag) / "report" / f"{attack_key}_report.json"
        if report_path.exists():
            try:
                with report_path.open("r", encoding="utf-8") as stream:
                    json.load(stream)
            except (OSError, json.JSONDecodeError):
                pass
            else:
                continue
        records = load_prompts_from_attack_json(file_path)
        records = deterministic_sample(
            records, 500, seed, f"test:{method_name(file_path)}"
        )
        if Path(file_path).name == "autodan_1.json":
            sequences = autodan_input_ids(detector.tokenizer, file_path, records)
            states = extract_input_ids_activations(
                detector.model,
                detector.tokenizer,
                sequences,
                detector.colon_batch_size,
                detector.device,
            )
            attack_scores, predicted_labels, _ = detector.predict_activations(
                {"colon": states, "ist": states}
            )
        else:
            prompts = [record["prompt"] for record in records]
            attack_scores, predicted_labels, _ = detector.predict_batch(prompts)
        results_detail = []
        for position, (record, attack_score, predicted_label) in enumerate(
            zip(records, attack_scores, predicted_labels), start=1
        ):
            prompt = record["prompt"]
            results_detail.append({
                "Sample_Index": position,
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "True_Label": record["true_label"],
                "Predicted_Label": int(predicted_label),
                "Anomaly_Score": round(-float(attack_score), 4),
                "Prediction_Result": "Jailbreak Prompt" if predicted_label else "Benign prompt",
            })
        generate_report(attack_key, results_detail, your_flag, len(records))
        print(f"evaluated {attack_key}: {len(records)}", flush=True)
    extract_accuracy_to_excel(your_flag)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO)
    set_determinism(args.seed)
    output_root = Path(args.output_dir).expanduser()
    your_flag = output_root / (
        f"seed-{args.seed}_"
        f"colon-k-{args.colon_k}-trees-{args.colon_n_estimators}-samples-{args.colon_max_samples}_"
        f"ist-k-{args.ist_k}-trees-{args.ist_n_estimators}-samples-{args.ist_max_samples}_"
        f"ist-weight-{IST_WEIGHT:.4f}"
    )
    colon_batch_size = args.colon_batch_size
    ist_batch_size = args.ist_batch_size
    rank_batch_size = 64
    colon_settings = {
        "k": args.colon_k,
        "n_estimators": args.colon_n_estimators,
        "max_samples": args.colon_max_samples,
    }
    ist_settings = {
        "k": args.ist_k,
        "n_estimators": args.ist_n_estimators,
        "max_samples": args.ist_max_samples,
    }

    benign_train_set_list = [
        ["datasets/train_data/databricks-dolly-15k.txt", 300],
        ["datasets/train_data/alpaca.txt", 300],
        ["datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200],
    ]
    malicious_train_set_list = [
        ["datasets/train_data/AdvBench.txt", 100],
        ["datasets/train_data/MaliciousInstruct.txt", 100],
        ["datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600],
    ]
    attack_dir = "datasets/vicuna_test"
    attack_file_path_list = [
        os.path.join(attack_dir, name)
        for name in sorted(os.listdir(attack_dir))
        if name.endswith(".json")
    ]
    model_path = "model/vicuna-7b-v1_5"
    model_dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": args.device},
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    configure_tokenizer(tokenizer)
    model.eval()

    benign_prompts, malicious_prompts = get_train_dataset(
        benign_train_set_list,
        malicious_train_set_list,
        args.seed,
    )
    background_layered_activations, all_labels = extract_trainset_hiddenstates(
        your_flag,
        args.device,
        tokenizer,
        model,
        benign_prompts,
        malicious_prompts,
        colon_batch_size,
        ist_batch_size,
    )
    detector = JailbreakDetector(
        model=model,
        tokenizer=tokenizer,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        your_flag=your_flag,
        colon_settings=colon_settings,
        ist_settings=ist_settings,
        ist_weight=IST_WEIGHT,
        random_state=args.seed,
        device=args.device,
        colon_batch_size=colon_batch_size,
        ist_batch_size=ist_batch_size,
        rank_batch_size=rank_batch_size,
    )
    eval(attack_file_path_list, detector, your_flag, args.seed)
    evaluate_attack_auroc(your_flag / "report", "vicuna_test")


if __name__ == "__main__":
    main()
