import argparse
import csv
import json
from pathlib import Path

import feature_protocol as sampling
from JailbreakDetector_llama2 import JailbreakDetector
from draw_auroc import evaluate_attack_auroc
from extract_trainset_hiddenstates_llama2 import ATTACK_FILES, extract_trainset_hiddenstates

HERE = Path(__file__).resolve().parent
PROJECT = HERE
MODEL = "llama2"
ATTACK_DIR = PROJECT / "datasets/llama2_test"


def parse_args(argv=None):
    selected = json.loads((HERE / "configs.json").read_text(encoding="utf-8"))[MODEL]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=selected["seed"])
    parser.add_argument("--k", type=int, default=selected["k"])
    parser.add_argument("--n-estimators", type=int, default=selected["n_estimators"])
    parser.add_argument("--max-samples", type=int, default=selected["max_samples"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / MODEL)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    args = parser.parse_args(argv)
    if min(args.k, args.n_estimators, args.max_samples, args.batch_size) < 1:
        parser.error("k, n-estimators, max-samples, and batch-size must be positive")
    return args


def list_available_attacks(attack_dir):
    paths = sorted(attack_dir / name for name in ATTACK_FILES)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("Selected attack data is missing")
    return paths


def load_prompts_from_attack_json(file_path, seed):
    records = sampling.load_records(file_path)
    return sampling.sample_records(records, seed, sampling.method_name(file_path))


def get_train_dataset(benign_path_list, malicious_path_list):
    return (
        [(PROJECT / path, count) for path, count in benign_path_list],
        [(PROJECT / path, count) for path, count in malicious_path_list],
    )


def predict(detector):
    return detector.score_all()


def eval(attack_file_path_list, detector, seed, report_dir):
    for path in report_dir.glob("*_results_detail.csv"):
        path.unlink()
    rows = detector.rows
    scores = predict(detector)
    datasets = [ATTACK_DIR / "toxic-chat_benign_0.json"] + attack_file_path_list
    for file_path in datasets:
        dataset = sampling.method_name(file_path)
        expected_indices = [
            row["source_index"] for row in load_prompts_from_attack_json(file_path, seed)
        ]
        selected = [
            (row, float(score)) for row, score in zip(rows, scores)
            if row["dataset"] == dataset
        ]
        if [row["source_index"] for row, _ in selected] != expected_indices:
            raise ValueError(f"Test selection mismatch: {dataset}")
        role = "benign" if file_path.name == "toxic-chat_benign_0.json" else "attack"
        if any(row["role"] != role for row, _ in selected):
            raise ValueError(f"Test label mismatch: {dataset}")
        suffix = "0" if role == "benign" else "1"
        output = report_dir / f"{MODEL}_test_{dataset}_{suffix}_results_detail.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Sample_Index", "Source_Index", "True_Label", "Predicted_Label", "Anomaly_Score"))
            for index, (row, score) in enumerate(selected, 1):
                writer.writerow((index, row["source_index"], int(role == "attack"), int(score < 0), score))
    evaluate_attack_auroc(str(report_dir), f"{MODEL}_test")
    with (report_dir / "all_attack_auroc_results.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        per_attack = {row["Attack Method"]: float(row["AUROC"]) for row in csv.DictReader(stream)}
    return sum(per_attack.values()) / len(per_attack), per_attack


def main(argv=None):
    args = parse_args(argv)
    setting = (
        f"seed-{args.seed}_k-{args.k}_"
        f"trees-{args.n_estimators}_samples-{args.max_samples}"
    )
    report_dir = args.output_dir.resolve() / setting / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    benign_train_set_list, malicious_train_set_list = get_train_dataset(
        [
            ("datasets/train_data/databricks-dolly-15k.txt", 300),
            ("datasets/train_data/alpaca.txt", 300),
            ("datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200),
        ],
        [
            ("datasets/train_data/AdvBench.txt", 100),
            ("datasets/train_data/MaliciousInstruct.txt", 100),
            ("datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600),
        ],
    )
    attack_file_path_list = list_available_attacks(ATTACK_DIR)
    rank_data = extract_trainset_hiddenstates(
        args.seed, args.device, args.dtype, args.batch_size,
        benign_train_set_list, malicious_train_set_list,
        args.rebuild_cache, args.force_rebuild_cache, k=args.k,
    )
    detector = JailbreakDetector(
        rank_data=rank_data,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        max_samples=args.max_samples,
        k_nb=args.k,
    )
    mean, per_attack = eval(attack_file_path_list, detector, args.seed, report_dir)
    print(f"{MODEL} seed {args.seed}: mean AUROC {mean:.16f}, SAA AUROC {per_attack['saa']:.6f}")
    print(f"AUROC report: {report_dir / 'all_attack_auroc_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
