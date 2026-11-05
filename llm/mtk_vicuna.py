from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Indicator_analysis_drawing import generate_report
from JailbreakDetector_vicuna import JailbreakDetector
from draw_auroc import evaluate_attack_auroc
from extract_AC_json import extract_accuracy_to_excel
from extract_trainset_hiddenstates_vicuna import (
    configure_tokenizer,
    extract_trainset_hiddenstates,
)


HERE = Path(__file__).resolve().parent
REPO = HERE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Vicuna BOS + final ANT (-1) single-endpoint MTK"
    )
    parser.add_argument("--seed", type=int, default=170)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=str(HERE / "results" / "vicuna"))
    args = parser.parse_args(argv)
    positive = (
        args.k,
        args.n_estimators,
        args.max_samples,
        args.batch_size,
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
    for path in (Path(your_flag) / "report").glob("vicuna_test_pap_*"):
        if path.is_file():
            path.unlink()
    for file_path in attack_file_path_list:
        attack_key = f"vicuna_test_{Path(file_path).stem}"
        records = load_prompts_from_attack_json(file_path)
        records = deterministic_sample(
            records, 500, seed, f"test:{method_name(file_path)}"
        )
        prompts = [record["prompt"] for record in records]
        attack_scores, predicted_labels, _ = detector.predict_batch(prompts)
        results_detail = []
        for position, (record, attack_score, predicted_label) in enumerate(
            zip(records, attack_scores, predicted_labels), start=1
        ):
            prompt = record["prompt"]
            results_detail.append({
                "Sample_Index": position,
                "Source_Index": record["source_index"],
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
    output_root = Path(args.output_dir).expanduser().resolve()
    os.chdir(REPO)
    torch.set_num_threads(2)
    set_determinism(args.seed)
    your_flag = output_root / f"seed-{args.seed}"
    rank_batch_size = 64
    settings = {
        "k": args.k,
        "n_estimators": args.n_estimators,
        "max_samples": args.max_samples,
    }

    benign_train_set_list = [
        ["datasets/reference_bank_data/databricks-dolly-15k.txt", 300],
        ["datasets/reference_bank_data/alpaca.txt", 300],
        ["datasets/reference_bank_data/non_refusal_prompts_with_responses_80k.txt", 200],
    ]
    malicious_train_set_list = [
        ["datasets/reference_bank_data/AdvBench.txt", 100],
        ["datasets/reference_bank_data/MaliciousInstruct.txt", 100],
        ["datasets/reference_bank_data/PKU-SafeRLHF-prompts_3-6k.txt", 600],
    ]
    attack_dir = "datasets/vicuna_test"
    attack_file_path_list = [
        os.path.join(attack_dir, name)
        for name in sorted(os.listdir(attack_dir))
        if name.endswith(".json")
    ]
    model_path = "model/vicuna_7b"
    model_dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": args.device},
        trust_remote_code=True,
        torch_dtype=model_dtype,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    configure_tokenizer(tokenizer)
    model.eval()

    benign_prompts, malicious_prompts = get_train_dataset(
        benign_train_set_list,
        malicious_train_set_list,
        args.seed,
    )
    background_layered_activations, all_labels = extract_trainset_hiddenstates(
        args.device,
        tokenizer,
        model,
        benign_prompts,
        malicious_prompts,
        args.batch_size,
    )
    detector = JailbreakDetector(
        model=model,
        tokenizer=tokenizer,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        settings=settings,
        random_state=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        rank_batch_size=rank_batch_size,
    )
    eval(attack_file_path_list, detector, your_flag, args.seed)
    per_attack = evaluate_attack_auroc(
        your_flag / "report",
        "vicuna_test",
    )
    mean_auroc = sum(per_attack.values()) / len(per_attack)
    summary = {
        "seed": args.seed,
        "input_protocol": "vicuna_bos_ant_nocolon_minus1_v1",
        "endpoint": "ANT (-1)",
        "settings": settings,
        "batch_size": args.batch_size,
        "rank_batch_size": rank_batch_size,
        "forest_implementation": "node_local_adjacent_midpoint_v1",
        "forest_stats": detector.forest_stats,
        "mean_auroc": mean_auroc,
        **per_attack,
        "test_data_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in attack_file_path_list},
    }
    (your_flag / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"vicuna seed {args.seed}: mean AUROC {mean_auroc:.6f}, "
        f"SAA AUROC {per_attack['SAA']:.6f}"
    )
    print(
        "AUROC report:",
        Path(your_flag) / "report" / "all_attack_auroc_results.csv",
    )


if __name__ == "__main__":
    main()
