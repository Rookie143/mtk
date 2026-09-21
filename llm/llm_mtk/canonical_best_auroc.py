from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent / "canonical_assets/llama2"
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonical_isolation_forest import PyTorchIsolationForest, _average_path_length

MODEL_PATH = REPO / "model/llama2"
TEST_DIR = REPO / "datasets/llama2_test"
BENIGN_TEST = TEST_DIR / "toxic-chat_benign_0.json"
TRAINING_PROFILE = {
    "benign": [
        (REPO / "datasets/train_data/databricks-dolly-15k.txt", 300),
        (REPO / "datasets/train_data/alpaca.txt", 300),
        (REPO / "datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200),
    ],
    "malicious": [
        (REPO / "datasets/train_data/AdvBench.txt", 100),
        (REPO / "datasets/train_data/MaliciousInstruct.txt", 100),
        (REPO / "datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt", 600),
    ],
}
K_VALUES = (1, 3, 5, 10, 15, 20, 25, 30)
FEATURE_ENDPOINT = "last_token_native_llama2_chat_template"


def stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        return [line for line in stream.readlines() if line.strip()]


def sample_training(seed: int):
    selected: dict[str, list[str]] = {"benign": [], "malicious": []}
    sources = []
    for role in ("benign", "malicious"):
        for path, count in TRAINING_PROFILE[role]:
            rows = read_lines(path)
            rng = random.Random(stable_seed(seed, f"train:{role}:{path.name}"))
            indices = rng.sample(range(len(rows)), count)
            selected[role].extend(rows[index] for index in indices)
            sources.append({
                "role": role, "path": str(path.resolve()), "sha256": sha256_file(path),
                "available": len(rows), "selected": count, "source_indices": indices,
            })
    return selected["benign"], selected["malicious"], sources


def method_name(path: Path) -> str:
    stem = path.stem
    return stem[:-2] if stem.endswith(("_0", "_1")) else stem


def attack_files() -> list[Path]:
    return [
        path for path in sorted(TEST_DIR.glob("*_1.json"))
        if "toxic-chat" not in path.name and "normal_" not in path.name
    ]


def load_records(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:

        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    output = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt = item.get("jailbreak") or item.get("prompt")
        if isinstance(prompt, str) and prompt:
            output.append({"source_index": index, "prompt": prompt})
    if not output:
        raise ValueError(f"No usable prompts in {path}")
    return output


def sample_records(records, seed: int, namespace: str, limit: int = 500):
    if len(records) <= limit:
        return records
    rng = random.Random(stable_seed(seed, f"test:{namespace}"))
    return [records[index] for index in rng.sample(range(len(records)), limit)]


def configure_tokenizer(tokenizer) -> None:
    dedicated_pad = "<|finetune_right_pad_id|>"
    if dedicated_pad in tokenizer.get_vocab():
        tokenizer.pad_token = dedicated_pad
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


def render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ) for prompt in prompts]


def extract_activations(model, tokenizer, prompts: list[str], batch_size: int, device: str):
    chunks = []
    max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        rendered = render_prompts(tokenizer, batch_prompts)
        encoded = tokenizer(
            rendered, padding=True, truncation=True, max_length=max_length,
            add_special_tokens=False, return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )

        states = torch.stack([layer[:, -1, :] for layer in outputs.hidden_states[1:]], dim=1)
        chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
        print(f"features {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(chunks, dim=0)


def model_and_tokenizer(device: str, dtype: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    configure_tokenizer(tokenizer)
    torch_dtype = getattr(torch, dtype)
    kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def feature_manifest(seed: int, sources, dtype: str):
    return {
        "seed": seed, "feature_endpoint": FEATURE_ENDPOINT,
        "dtype": dtype, "training_sources": sources,
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
    }


def prepare_training_features(seed: int, device: str, dtype: str, batch_size: int, force=False,
                              model=None, tokenizer=None):
    output = HERE / "cache" / "training" / f"seed-{seed}.pt"
    benign, malicious, sources = sample_training(seed)
    manifest = feature_manifest(seed, sources, dtype)
    if output.is_file() and not force:
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("manifest") == manifest:
            print(f"seed {seed}: validated feature cache", flush=True)
            return output
    if model is None or tokenizer is None:
        model, tokenizer = model_and_tokenizer(device, dtype)
    features = extract_activations(model, tokenizer, benign + malicious, batch_size, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "manifest": manifest, "features": features,
        "labels": torch.tensor([0] * len(benign) + [1] * len(malicious)),
    }, output)
    return output


def test_cache_manifest(dtype: str):
    paths = [BENIGN_TEST] + attack_files()
    return {
        "feature_endpoint": FEATURE_ENDPOINT, "dtype": dtype,
        "files": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in paths],
        "model_config_sha256": sha256_file(MODEL_PATH / "config.json"),
        "tokenizer_config_sha256": sha256_file(MODEL_PATH / "tokenizer_config.json"),
    }


def prepare_test_features(device: str, dtype: str, batch_size: int, force=False,
                          model=None, tokenizer=None):
    output = HERE / "cache" / "test_features.pt"
    manifest = test_cache_manifest(dtype)
    if output.is_file() and not force:
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("manifest") == manifest:
            print("validated test feature cache", flush=True)
            return output
    datasets = {}
    all_prompts = []
    for path in [BENIGN_TEST] + attack_files():
        records = load_records(path)
        start = len(all_prompts)
        all_prompts.extend(row["prompt"] for row in records)
        datasets[method_name(path)] = {
            "path": str(path.resolve()), "start": start, "count": len(records),
            "source_indices": [row["source_index"] for row in records],
        }
    if model is None or tokenizer is None:
        model, tokenizer = model_and_tokenizer(device, dtype)
    features = extract_activations(model, tokenizer, all_prompts, batch_size, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "datasets": datasets, "features": features}, output)
    return output


def selected_test(seed: int, saved):
    feature_parts, rows = [], []
    ordered = [method_name(BENIGN_TEST)] + [method_name(p) for p in attack_files()]
    for dataset in ordered:
        info = saved["datasets"][dataset]
        records = [{"local_index": i, "source_index": source}
                   for i, source in enumerate(info["source_indices"])]
        chosen = sample_records(records, seed, dataset, 500)
        indices = torch.tensor([info["start"] + row["local_index"] for row in chosen])
        feature_parts.append(saved["features"][indices])
        role = "benign" if dataset == method_name(BENIGN_TEST) else "attack"
        rows.extend({"dataset": dataset, "role": role, "source_index": row["source_index"]}
                    for row in chosen)
    return torch.cat(feature_parts), rows


def rank_features(queries, background, labels, k_values, device: str, exclude_self=False, batch_size=64):
    n_queries, n_layers, _ = queries.shape
    output = {k: torch.empty((n_queries, n_layers), dtype=torch.float32) for k in k_values}

    background = background.to(device=device)
    labels = labels.to(device)
    benign_count = int((labels == 0).sum()) - (1 if exclude_self else 0)
    if max(k_values) > benign_count:
        raise ValueError(f"k={max(k_values)} exceeds available benign references {benign_count}")
    positions = torch.arange(1, len(labels) + 1, device=device, dtype=torch.float32)
    for layer in range(n_layers):
        refs = background[:, layer, :]
        for start in range(0, n_queries, batch_size):
            stop = min(start + batch_size, n_queries)
            query = queries[start:stop, layer, :].to(device=device, dtype=background.dtype)
            distances = (refs.unsqueeze(0) - query.unsqueeze(1)).norm(p=2, dim=2)
            if exclude_self:
                local = torch.arange(stop - start, device=device)
                distances[local, torch.arange(start, stop, device=device)] = torch.inf
            order = distances.argsort(dim=1)
            benign_sorted = labels[order].eq(0)
            ranked_positions = positions.expand(stop - start, -1).masked_fill(~benign_sorted, torch.inf)
            nearest_positions = ranked_positions.topk(max(k_values), largest=False, dim=1).values
            cumulative = nearest_positions.cumsum(dim=1)
            for k in k_values:
                output[k][start:stop, layer] = (cumulative[:, k - 1] / k).cpu()
        print(f"ranks layer {layer + 1}/{n_layers}", flush=True)
    return output


def prepare_ranks(seed: int, device: str, force=False):
    output = HERE / "cache" / "ranks" / f"seed-{seed}.pt"
    train_path = HERE / "cache" / "training" / f"seed-{seed}.pt"
    test_path = HERE / "cache" / "test_features.pt"
    train = torch.load(train_path, map_location="cpu", weights_only=False)
    test = torch.load(test_path, map_location="cpu", weights_only=False)
    selected_features, rows = selected_test(seed, test)
    manifest = {
        "seed": seed, "k_values": list(K_VALUES),
        "training_manifest": train["manifest"], "test_manifest": test["manifest"],
        "sample_rows": rows,
    }
    if output.is_file() and not force:
        saved = torch.load(output, map_location="cpu", weights_only=False)
        if saved.get("manifest") == manifest:
            print(f"seed {seed}: validated rank cache", flush=True)
            return output
    train_ranks = rank_features(
        train["features"], train["features"], train["labels"], K_VALUES,
        device, exclude_self=True,
    )
    test_ranks = rank_features(
        selected_features, train["features"], train["labels"], K_VALUES,
        device, exclude_self=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"manifest": manifest, "labels": train["labels"],
                "train_ranks": train_ranks, "test_ranks": test_ranks}, output)
    return output


def evaluate_setting(seed: int, k: int, n_estimators: int, max_samples: int, save_predictions=False):
    saved = torch.load(HERE / "cache/ranks" / f"seed-{seed}.pt", map_location="cpu", weights_only=False)
    labels = saved["labels"]
    benign_train = saved["train_ranks"][k][labels == 0]
    mean = benign_train.mean(dim=0, keepdim=True)
    std = benign_train.std(dim=0, keepdim=True) + 1e-8
    normalized_train = (benign_train - mean) / std
    normalized_test = (saved["test_ranks"][k] - mean) / std
    forest = PyTorchIsolationForest(
        n_estimators=n_estimators, max_samples=max_samples, random_state=seed
    ).fit(normalized_train)
    decision = forest.decision_function(normalized_test).detach().cpu().numpy()
    rows = saved["manifest"]["sample_rows"]
    return summarize_scores(seed, k, n_estimators, max_samples, decision, rows, save_predictions)


def summarize_scores(seed, k, n_estimators, max_samples, decision, rows, save_predictions=False):
    benign_scores = -decision[[row["role"] == "benign" for row in rows]]
    aurocs = {}
    for dataset in sorted({row["dataset"] for row in rows if row["role"] == "attack"}):
        attack_scores = -decision[[row["dataset"] == dataset for row in rows]]
        truth = np.r_[np.zeros(len(benign_scores)), np.ones(len(attack_scores))]
        aurocs[dataset] = float(roc_auc_score(truth, np.r_[benign_scores, attack_scores]))
    result = {
        "seed": seed, "k": k, "n_estimators": n_estimators, "max_samples": max_samples,
        "mean_auroc": float(np.mean(list(aurocs.values()))), "per_attack_auroc": aurocs,
        "sample_counts": {dataset: sum(row["dataset"] == dataset for row in rows)
                          for dataset in sorted({row["dataset"] for row in rows})},
    }
    if save_predictions:
        result["prediction_rows"] = [
            {**row, "true_label": int(row["role"] == "attack"),
             "decision_score": float(score), "attack_score": float(-score)}
            for row, score in zip(rows, decision)
        ]
    return result


def evaluate_tree_prefixes(seed: int, k: int, max_samples: int,
                           tree_counts=(100, 300, 500, 800)):
    saved = torch.load(HERE / "cache/ranks" / f"seed-{seed}.pt", map_location="cpu", weights_only=False)
    labels = saved["labels"]
    benign_train = saved["train_ranks"][k][labels == 0]
    mean = benign_train.mean(dim=0, keepdim=True)
    std = benign_train.std(dim=0, keepdim=True) + 1e-8
    train = (benign_train - mean) / std
    test = (saved["test_ranks"][k] - mean) / std
    forest = PyTorchIsolationForest(
        n_estimators=max(tree_counts), max_samples=max_samples, random_state=seed
    ).fit(train)
    compiled = forest._compiled_trees
    tree_count = max(tree_counts)
    batch = test.shape[0]
    tree_indices = torch.arange(tree_count).unsqueeze(0).expand(batch, -1)
    nodes = torch.zeros((batch, tree_count), dtype=torch.long)
    for _ in range(compiled["max_depth"] + 1):
        active = ~compiled["is_leaf"][tree_indices, nodes]
        if not bool(active.any()):
            break
        access_features = compiled["feature"][tree_indices, nodes]
        values = test.gather(1, access_features)
        go_left = values < compiled["threshold"][tree_indices, nodes]
        next_nodes = torch.where(
            go_left, compiled["left"][tree_indices, nodes], compiled["right"][tree_indices, nodes]
        )
        nodes = torch.where(active, next_nodes, nodes)
    per_tree_depth = compiled["depth"][tree_indices, nodes] + compiled["leaf_extra"][tree_indices, nodes]
    cumulative = per_tree_depth.cumsum(dim=1)
    normalizer = _average_path_length(forest.max_samples_)[0]
    rows = saved["manifest"]["sample_rows"]
    results = []
    for count in tree_counts:
        normalized = cumulative[:, count - 1] / (count * normalizer)
        decision = (-(2.0 ** (-normalized)) + 0.5).numpy()
        results.append(summarize_scores(seed, k, count, max_samples, decision, rows))
    return results


def save_best_artifacts(root: Path, result) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "best_result.json", {k: v for k, v in result.items() if k != "prediction_rows"})
    rows = result.get("prediction_rows", [])
    if rows:
        with (root / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader();
            writer.writerows(rows)


def command_prepare(args):
    model, tokenizer = model_and_tokenizer(args.device, args.dtype)
    prepare_test_features(args.device, args.dtype, args.batch_size, args.force, model, tokenizer)
    for seed in args.seeds:
        prepare_training_features(seed, args.device, args.dtype, args.batch_size, args.force, model, tokenizer)
        prepare_ranks(seed, args.device, args.force)


def command_seed_search(args):
    root = HERE / "seed_only"
    records = []
    for seed in args.seeds:
        result = evaluate_setting(seed, 10, 500, 512)
        records.append(result)
        write_json(root / "trials" / f"seed-{seed}.json", result)
        print(f"seed={seed} mean_AUROC={result['mean_auroc']:.10f}", flush=True)
    ordered = sorted(records, key=lambda row: row["seed"])
    best = max(ordered, key=lambda row: (row["mean_auroc"], -row["seed"]))
    write_json(root / "all_results.json", {"protocol": "seed only", "trials": ordered, "best": best})
    save_best_artifacts(root / "best", evaluate_setting(best["seed"], 10, 500, 512, True))


def command_hyper_search(args):
    seed_results = json.loads((HERE / "seed_only/all_results.json").read_text(encoding="utf-8"))["trials"]
    top_seeds = [row["seed"] for row in sorted(seed_results, key=lambda r: (-r["mean_auroc"], r["seed"]))[:5]]
    root = HERE / "hyperparameter_search"
    records, best = [], None
    for seed in top_seeds:
        for k in K_VALUES:
            for samples in (128, 256, 512, 800):
                for result in evaluate_tree_prefixes(seed, k, samples):
                    trees = result["n_estimators"]
                    records.append(result)
                    if best is None or (result["mean_auroc"], -trees, -k, -seed) > (
                    best["mean_auroc"], -best["n_estimators"], -best["k"], -best["seed"]):
                        best = result
                    write_json(root / "progress.json",
                               {"top_seeds": top_seeds, "completed": len(records), "best": best})
                    print(
                        f"trial={len(records)}/640 seed={seed} k={k} trees={trees} samples={samples} mean={result['mean_auroc']:.10f}",
                        flush=True)
    write_json(root / "all_results.json",
               {"protocol": "joint grid", "top_seeds": top_seeds, "trials": records, "best": best})
    save_best_artifacts(root / "best",
                        evaluate_setting(best["seed"], best["k"], best["n_estimators"], best["max_samples"], True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "seed-search", "hyper-search"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    set_determinism(42)
    {"prepare": command_prepare, "seed-search": command_seed_search,
     "hyper-search": command_hyper_search}[args.command](args)


if __name__ == "__main__":
    main()
