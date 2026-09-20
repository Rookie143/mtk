
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


TABLE_DIR = Path(__file__).resolve().parent
REPO_DIR = TABLE_DIR.parent
ASSET_DIR = TABLE_DIR / "canonical_assets"
SETTINGS_PATH = TABLE_DIR / "all_models_best_settings_20260915.json"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import canonical_best_auroc as base
import canonical_model_protocols as optimized


MODEL_LAYOUT = {
    "llama2": {
        "protocol_root": ASSET_DIR / "llama2",
        "model_path": REPO_DIR / "model/llama2",
        "test_dir": REPO_DIR / "datasets/llama2_test",
        "benign_file": "toxic-chat_benign_0.json",
        "feature_endpoint": "last_token_native_llama2_chat_template",
    },
    "llama3": {
        "protocol_root": ASSET_DIR / "llama3",
        "model_path": REPO_DIR / "model/llama3",
        "test_dir": REPO_DIR / "datasets/llama3_test",
        "benign_file": "toxic-chat_benign_0.json",
        "feature_endpoint": "last_token_native_llama3_chat_template",
    },
    "mistral": {
        "protocol_root": ASSET_DIR / "mistral",
        "model_path": REPO_DIR / "model/mistral_7b",
        "test_dir": REPO_DIR / "datasets/mistral_test",
        "benign_file": "toxic-chat_benign_0.json",
        "feature_endpoint": "mistral_slash_token_embedding_plus_layers_1_31_project_compatible",
    },
}

_BASE_EXTRACT_ACTIVATIONS = base.extract_activations


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _llama_attack_files():
    return [
        path
        for path in sorted(base.TEST_DIR.glob("*_1.json"))
        if "toxic-chat" not in path.name.lower()
        and "normal_" not in path.name.lower()
    ]


def configure_protocol(model: str) -> dict:

    layout = MODEL_LAYOUT[model]
    if model == "mistral":
        optimized.configure("mistral")
        return layout

    base.HERE = layout["protocol_root"]
    base.MODEL_PATH = layout["model_path"]
    base.TEST_DIR = layout["test_dir"]
    base.BENIGN_TEST = layout["test_dir"] / layout["benign_file"]
    base.FEATURE_ENDPOINT = layout["feature_endpoint"]
    base.extract_activations = _BASE_EXTRACT_ACTIVATIONS
    base.attack_files = _llama_attack_files
    return layout


def rebuild_rank_cache(
    model: str,
    seed: int,
    device: str,
    dtype: str,
    batch_size: int,
    force: bool,
) -> None:

    if model == "mistral":
        args = SimpleNamespace(
            model="mistral",
            seeds=[seed],
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            force=force,
        )
        optimized.prepare(args)
        return

    model_object, tokenizer = base.model_and_tokenizer(device, dtype)
    base.prepare_test_features(
        device,
        dtype,
        batch_size,
        force,
        model=model_object,
        tokenizer=tokenizer,
    )
    base.prepare_training_features(
        seed,
        device,
        dtype,
        batch_size,
        force,
        model=model_object,
        tokenizer=tokenizer,
    )
    base.prepare_ranks(seed, device, force)


def recorded_result(model: str, settings: dict) -> tuple[Path, dict]:
    source = REPO_DIR / settings[model]["source"]
    return source, load_json(source)


def selected_parameters(model: str, args) -> dict:
    selected = load_json(SETTINGS_PATH)[model]
    return {
        "seed": selected["seed"] if args.seed is None else args.seed,
        "k": selected["k"] if args.k is None else args.k,
        "n_estimators": (
            selected["n_estimators"]
            if args.n_estimators is None
            else args.n_estimators
        ),
        "max_samples": (
            selected["max_samples"]
            if args.max_samples is None
            else args.max_samples
        ),
    }


def verify_result(actual: dict, expected: dict, tolerance: float) -> dict:
    differences = {
        name: actual["per_attack_auroc"].get(name, float("nan")) - value
        for name, value in expected["per_attack_auroc"].items()
    }
    missing = sorted(set(expected["per_attack_auroc"]) - set(actual["per_attack_auroc"]))
    unexpected = sorted(set(actual["per_attack_auroc"]) - set(expected["per_attack_auroc"]))
    mean_difference = actual["mean_auroc"] - expected["mean_auroc"]
    passed = (
        not missing
        and not unexpected
        and abs(mean_difference) <= tolerance
        and all(abs(value) <= tolerance for value in differences.values())
    )
    return {
        "passed": passed,
        "absolute_tolerance": tolerance,
        "mean_difference": mean_difference,
        "per_attack_differences": differences,
        "missing_attacks": missing,
        "unexpected_attacks": unexpected,
    }


def write_outputs(
    output_dir: Path,
    model: str,
    result: dict,
    expected_path: Path,
    expected: dict,
    rank_path: Path,
    verification: dict | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "prediction_rows"}
    write_json(output_dir / "result.json", summary)

    with (output_dir / "all_attack_auroc_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["Attack Method", "AUROC"])
        writer.writerows(sorted(result["per_attack_auroc"].items()))

    prediction_rows = result.get("prediction_rows", [])
    if prediction_rows:
        with (output_dir / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)

    manifest = {
        "model": model,
        "protocol": "canonical local rank-cache evaluation",
        "protocol_implementation": str((TABLE_DIR / "canonical_best_auroc.py").resolve()),
        "model_protocol_implementation": str((TABLE_DIR / "canonical_model_protocols.py").resolve()),
        "isolation_forest_implementation": str(
            (TABLE_DIR / "canonical_isolation_forest.py").resolve()
        ),
        "rank_cache": str(rank_path.resolve()),
        "rank_cache_sha256": sha256_file(rank_path),
        "rank_manifest": __import__("torch").load(
            rank_path, map_location="cpu", weights_only=False
        )["manifest"],
        "recorded_result": str(expected_path.resolve()),
        "recorded_mean_auroc": expected["mean_auroc"],
        "computed_mean_auroc": result["mean_auroc"],
        "verification": verification,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if verification is not None:
        write_json(output_dir / "verification.json", verification)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce a selected canonical Table 4 result"
    )
    parser.add_argument("model", choices=sorted(MODEL_LAYOUT))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        help="Base output directory; a setting-specific directory is created below it",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Run the canonical raw-prompt feature/rank preparation before evaluation",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Regenerate canonical feature/rank caches even when validated caches exist",
    )
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    all_settings = load_json(SETTINGS_PATH)
    params = selected_parameters(args.model, args)
    layout = configure_protocol(args.model)

    rank_path = layout["protocol_root"] / "cache/ranks" / f"seed-{params['seed']}.pt"
    if args.force_rebuild_cache or args.rebuild_cache or not rank_path.is_file():
        rebuild_rank_cache(
            args.model,
            params["seed"],
            args.device,
            args.dtype,
            args.batch_size,
            args.force_rebuild_cache,
        )
    if not rank_path.is_file():
        raise FileNotFoundError(f"Canonical rank cache was not produced: {rank_path}")

    base.set_determinism(42)
    result = base.evaluate_setting(
        params["seed"],
        params["k"],
        params["n_estimators"],
        params["max_samples"],
        save_predictions=True,
    )

    expected_path, expected = recorded_result(args.model, all_settings)
    exact_selected = all(
        params[key] == all_settings[args.model][key]
        for key in ("seed", "k", "n_estimators", "max_samples")
    )
    verification = None
    if not args.no_verify and exact_selected:
        verification = verify_result(result, expected, args.tolerance)

    output_base = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else TABLE_DIR / "runs_best_auroc" / args.model
    )
    setting_name = (
        f"seed-{params['seed']}_k-{params['k']}_"
        f"trees-{params['n_estimators']}_samples-{params['max_samples']}"
    )
    output_dir = output_base / setting_name
    write_outputs(
        output_dir,
        args.model,
        result,
        expected_path,
        expected,
        rank_path,
        verification,
    )

    print(json.dumps({
        "model": args.model,
        **params,
        "mean_auroc": result["mean_auroc"],
        "recorded_mean_auroc": expected["mean_auroc"] if exact_selected else None,
        "verification_passed": verification["passed"] if verification else None,
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))

    if verification is not None and not verification["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
