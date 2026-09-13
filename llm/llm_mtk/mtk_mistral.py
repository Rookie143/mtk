import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIXED_CODE_DIR = Path(__file__).resolve().parent
if str(FIXED_CODE_DIR) in sys.path:
    sys.path.remove(str(FIXED_CODE_DIR))
sys.path.insert(0, str(FIXED_CODE_DIR))

from JailbreakDetector_mistral import JailbreakDetector
from Indicator_analysis_drawing import *
from draw_auroc import evaluate_attack_auroc
from extract_AC_json import extract_accuracy_to_excel
from extract_trainset_hiddenstates_mistral import extract_trainset_hiddenstates


REFERENCE_BANK_SEED = 2
FOREST_RANDOM_STATE = 42
TEST_LIMIT = 500


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def method_name(file_path: str) -> str:
    stem = Path(file_path).stem
    return stem[:-2] if stem.endswith(("_0", "_1")) else stem


def list_available_attacks(attack_dir):
    if not os.path.isdir(attack_dir):
        return []
    files = [f for f in os.listdir(attack_dir) if f.lower().endswith(".json")]
    return [os.path.splitext(f)[0] for f in sorted(files)]


def load_prompts_from_attack_json(file_path: str):
    prompts = []
    true_label = int(os.path.basename(file_path)[-6])
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as stream:
            data = json.load(stream)
    except json.JSONDecodeError:
        with open(file_path, "r", encoding="gbk", errors="ignore") as stream:
            data = json.load(stream)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} is not a JSON list.")
    for source_index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = item.get("jailbreak")
        if prompt and isinstance(prompt, str):
            prompts.append({
                "prompt": prompt,
                "true_label": true_label,
                "source_index": source_index,
            })
    return prompts


def deterministic_sample(rows, count: int, namespace: str):
    if len(rows) <= count:
        return rows
    rng = random.Random(stable_seed(REFERENCE_BANK_SEED, namespace))
    return [rows[index] for index in rng.sample(range(len(rows)), count)]


def get_train_dataset(benign_path_list, malicious_path_list):
    benign_prompts = []
    malicious_prompts = []
    for role, path_list, output in (
        ("benign", benign_path_list, benign_prompts),
        ("malicious", malicious_path_list, malicious_prompts),
    ):
        for path, count in path_list:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as stream:
                rows = [line for line in stream.readlines() if line.strip()]
            rng = random.Random(stable_seed(
                REFERENCE_BANK_SEED, f"train:{role}:{Path(path).name}"
            ))
            indices = rng.sample(range(len(rows)), min(count, len(rows)))
            output.extend(rows[index] for index in indices)
    return benign_prompts, malicious_prompts


def eval(attack_file_path_list, detector, your_flag):
    def get_last_two_levels(path):
        normalized_path = os.path.normpath(path)
        path_parts = normalized_path.split(os.sep)
        last_two_parts = path_parts[-2:] if len(path_parts) >= 2 else path_parts
        dir_name = last_two_parts[0]
        file_name = last_two_parts[1] if len(last_two_parts) > 1 else ""
        return f"{dir_name}_{os.path.splitext(file_name)[0]}"

    def is_already_detected(attack_key):
        report_file = os.path.join(f"./{your_flag}/report", f"{attack_key}_report.json")
        return os.path.exists(report_file)

    for attack_file_path in tqdm(attack_file_path_list, desc="Evaluating attack types"):
        attack_key = get_last_two_levels(attack_file_path)
        if is_already_detected(attack_key):
            continue

        test_samples = load_prompts_from_attack_json(attack_file_path)
        if not test_samples:
            continue
        test_samples = deterministic_sample(
            test_samples, TEST_LIMIT, f"test:{method_name(attack_file_path)}"
        )
        results_detail = []

        for idx, sample in enumerate(tqdm(test_samples, desc="Predicting samples")):
            pred_label_str, pred_label, anomaly_score = detector.predict(
                prompt_text=sample["prompt"], return_score=True
            )
            prompt = sample["prompt"]
            results_detail.append({
                "Sample_Index": idx + 1,
                "Source_Index": sample["source_index"],
                "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                "True_Label": sample["true_label"],
                "Predicted_Label": pred_label,
                "Anomaly_Score": float(anomaly_score),
                "Prediction_Result": pred_label_str,
            })

        generate_report(attack_key, results_detail, your_flag, len(test_samples))
    extract_accuracy_to_excel(your_flag)


def main():
    start_time = time.time()
    os.chdir(REPO_ROOT)
    your_flag = "best_auroc/mistral_project_compatible_no_target_leak/output"
    ab_k = 10
    n_esti = 500
    max_samp = 512
    now_metric = "l2"
    target_layers_indices = list(range(0, 32))

    benign_train_set_list = [
        ["./datasets/train_data/databricks-dolly-15k.txt", 300],
        ["./datasets/train_data/alpaca.txt", 300],
        ["./datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200],
    ]
    malicious_train_set_list = [
        ["./datasets/train_data/AdvBench.txt", 100],
        ["./datasets/train_data/MaliciousInstruct.txt", 100],
        ["./datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600],
    ]

    attack_dir = "./datasets/mistral_test/"
    attack_file_path_list = [
        os.path.join(attack_dir, name)
        for name in sorted(os.listdir(attack_dir))
        if name.lower().endswith(".json")
    ]

    model_path = "./model/mistral_7b/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benign_prompts, malicious_prompts = get_train_dataset(
        benign_train_set_list, malicious_train_set_list
    )
    background_layered_activations, all_labels = extract_trainset_hiddenstates(
        your_flag, device, tokenizer, model, benign_prompts, malicious_prompts
    )
    background_layered_activations = background_layered_activations[:, target_layers_indices, :]

    detector = JailbreakDetector(
        model=model,
        tokenizer=tokenizer,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        your_flag=your_flag,
        n_estimators=n_esti,
        random_state=FOREST_RANDOM_STATE,
        max_samples=max_samp,
        k_nb=ab_k,
        target_layers=target_layers_indices,
        metric=now_metric,
    )
    eval(attack_file_path_list, detector, your_flag)
    evaluate_attack_auroc(f"{your_flag}/report/", "mistral_test")
    print(f"Finished in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
