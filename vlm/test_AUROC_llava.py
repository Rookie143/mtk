import argparse
import csv
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, precision_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

from JailbreakDetector_llava import JailbreakDetector
from load_datasets import (
    load_FigImg,
    load_JailBreakV_JBtxt_SDimg,
    load_mm_safety_bench_all,
    load_mm_vet_v2,
    load_sd_advbench_for_train,
    load_usb_datasset,
    load_usb_datasset_for_train,
    load_vqa_dataset_for_train,
)


MODEL_PREFIX = "llava"


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_threshold_for_target_fpr(benign_scores, target_fpr=0.05):
    benign_scores = np.asarray(benign_scores)
    sorted_scores = np.sort(benign_scores)[::-1]
    n_positive = int(np.ceil(len(sorted_scores) * target_fpr))
    if n_positive == 0:
        threshold = sorted_scores[0] + 1e-6
    elif n_positive >= len(sorted_scores):
        threshold = sorted_scores[-1] - 1e-6
    else:
        threshold = sorted_scores[n_positive - 1]
    actual_fpr = (benign_scores >= threshold).sum() / len(benign_scores)
    print(f"Target FPR={target_fpr}, threshold={threshold}, actual FPR={actual_fpr:.4f}")
    return threshold


def calculate_metrics_with_threshold(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred_labels = (scores >= threshold).astype(int)
    n_negative = (labels == 0).sum()
    false_positive = ((pred_labels == 1) & (labels == 0)).sum()
    fpr = false_positive / n_negative if n_negative > 0 else 0.0
    precision = precision_score(labels, pred_labels, zero_division=0.0)
    print(f"FPR: {fpr:.4f}")
    print(f"Precision: {precision:.4f}")
    return fpr, precision


def evaluate_auprc(true_labels, scores):
    precision_arr, recall_arr, _ = precision_recall_curve(true_labels, scores)
    return auc(recall_arr, precision_arr)


def evaluate_auroc(true_labels, scores):
    fpr, tpr, _ = roc_curve(true_labels, scores)
    return auc(fpr, tpr)


def extract_features(model, processor, samples, device, desc):
    all_activations = []
    for sentence in tqdm(samples, desc=desc):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": sentence[1]},
                    {"type": "text", "text": sentence[0]},
                ],
            },
        ]
        with torch.no_grad():
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
            activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
            del inputs, outputs, hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        all_activations.append(activations)
    return all_activations


def load_or_build_feature_bank(args, model, processor, device):
    feature_path = args.output_dir / "saved_features_and_labels.pt"
    if feature_path.exists():
        loaded_data = torch.load(feature_path, map_location=device)
        return loaded_data["background_layered_activations"], loaded_data["labels"]

    benign_prompts = load_vqa_dataset_for_train() + load_usb_datasset_for_train()
    malicious_prompts = load_sd_advbench_for_train()
    benign_activations = extract_features(
        model, processor, benign_prompts, device, "Extract features of benign samples"
    )
    malicious_activations = extract_features(
        model, processor, malicious_prompts, device, "Extract features of malicious samples"
    )
    all_activations = benign_activations + malicious_activations
    benign_labels = torch.zeros(len(benign_prompts), device=device)
    malicious_labels = torch.ones(len(malicious_prompts), device=device)
    all_labels = torch.cat([benign_labels, malicious_labels], dim=0)

    num_layers = len(all_activations[0])
    layered_activations = []
    for layer_idx in range(num_layers):
        layer_feats = torch.stack([sample_feats[layer_idx] for sample_feats in all_activations], dim=0)
        layered_activations.append(layer_feats)
    background_layered_activations = torch.stack(layered_activations, dim=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "background_layered_activations": background_layered_activations,
            "labels": all_labels,
        },
        feature_path,
    )
    return background_layered_activations, all_labels


def evaluate_datasets(detector, args):
    datasets = {
        "MM-SafetyBench+MM-Vet": load_mm_safety_bench_all() + load_mm_vet_v2(),
        "FigImg+MM-Vet": load_FigImg() + load_mm_vet_v2(),
        "JBV28K_JBtxt_SDimg+MM-Vet": load_JailBreakV_JBtxt_SDimg() + load_mm_vet_v2(),
        "MM-SafetyBench+usb": load_mm_safety_bench_all() + load_usb_datasset(),
        "MM-Vet_all": load_mm_vet_v2(is_all=True),
    }
    results = {}
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting evaluation of {len(datasets)} datasets...")
    for idx, (dataset_name, dataset) in enumerate(datasets.items(), 1):
        print(f"Processing dataset {idx}/{len(datasets)}: {dataset_name}")
        true_labels = []
        scores = []
        for item in tqdm(dataset, desc="Processing dataset", leave=True, ncols=100):
            _, score = detector.predict([item["txt"], item["img"]])
            true_labels.append(item["toxicity"])
            scores.append(score)

        csv_path = results_dir / f"test_{MODEL_PREFIX}_{dataset_name}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["txt", "img", "true_labels", "score"])
            writer.writerows(
                zip(
                    [item["txt"] for item in dataset],
                    [item["img"] for item in dataset],
                    [item["toxicity"] for item in dataset],
                    scores,
                )
            )

        if dataset_name != "MM-Vet_all":
            auprc = evaluate_auprc(true_labels, scores)
            auroc = evaluate_auroc(true_labels, scores)
            results[dataset_name] = (auprc, auroc)
            print(f"AUPRC for {dataset_name}: {auprc}")
            print(f"AUROC for {dataset_name}: {auroc}")

        with open(results_dir / f"{MODEL_PREFIX}_AUROC_result.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Dataset Name", "AUPRC", "AUROC"])
            for name, result in results.items():
                writer.writerow([name, f"{result[0]:.4f}", f"{result[1]:.4f}"])

    benign_df = pd.read_csv(results_dir / f"test_{MODEL_PREFIX}_MM-Vet_all.csv")
    test_df = pd.read_csv(results_dir / f"test_{MODEL_PREFIX}_MM-SafetyBench+usb.csv")
    threshold = find_threshold_for_target_fpr(
        benign_df[benign_df["true_labels"] == 0]["score"].values,
        target_fpr=args.target_fpr,
    )
    calculate_metrics_with_threshold(test_df["score"].values, test_df["true_labels"].values, threshold)


def parse_args():
    default_run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Evaluate MTK VLM detector with LLaVA.")
    parser.add_argument("run_name", nargs="?", default=default_run_name)
    parser.add_argument("--model-name-or-path", default="llava-hf/llava-v1.6-vicuna-7b-hf")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--k-nb", type=int, default=10)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    args = parser.parse_args()
    args.output_dir = args.output_dir or Path("experimental_results") / args.run_name
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch_dtype = getattr(torch, args.torch_dtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name_or_path,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )
    device = model_device(model)
    background_layered_activations, all_labels = load_or_build_feature_bank(args, model, processor, device)
    detector = JailbreakDetector(
        model=model,
        processor=processor,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        flag=args.run_name,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        k_nb=args.k_nb,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
    )
    evaluate_datasets(detector, args)


if __name__ == "__main__":
    main()
