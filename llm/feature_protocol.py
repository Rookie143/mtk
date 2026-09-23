from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


HERE = Path(__file__).resolve().parent
PROJECT = HERE


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


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        return [line for line in stream.readlines() if line.strip()]


def method_name(path: Path) -> str:
    stem = path.stem
    return stem[:-2] if stem.endswith(("_0", "_1")) else stem


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


def render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def native_extract(model, tokenizer, prompts: list[str], batch_size: int, device: str):
    chunks = []
    max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    for start in range(0, len(prompts), batch_size):
        rendered = render_prompts(tokenizer, prompts[start:start + batch_size])
        encoded = tokenizer(
            rendered, padding=True, truncation=True, max_length=max_length,
            add_special_tokens=False, return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        states = torch.stack([layer[:, -1, :] for layer in outputs.hidden_states[1:]], dim=1)
        chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
    return torch.cat(chunks, dim=0)


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
    return output


class FeatureProtocol:
    def __init__(
        self, model_name: str, model_dir: str, attack_files,
        benign_train_set_list, malicious_train_set_list,
        training_endpoint: str, test_endpoint: str,
        training_line_reader=None,
        rank_k_values=None,
    ):
        self.model_name = model_name
        self.cache_root = HERE / "canonical_assets" / model_name
        self.model_path = PROJECT / "model" / model_dir
        self.test_dir = PROJECT / "datasets" / f"{model_name}_test"
        self.benign_test = self.test_dir / "toxic-chat_benign_0.json"
        self.attack_files = sorted(self.test_dir / name for name in attack_files)
        self.training_profile = {
            "benign": [(Path(path), count) for path, count in benign_train_set_list],
            "malicious": [(Path(path), count) for path, count in malicious_train_set_list],
        }
        self.training_endpoint = training_endpoint
        self.test_endpoint = test_endpoint
        self.training_line_reader = training_line_reader or read_lines
        self.rank_k_values = tuple(rank_k_values) if rank_k_values is not None else None
        missing = [str(path) for path in [self.benign_test] + self.attack_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Required test data missing: {missing}")

    def model_and_tokenizer(self, device: str, dtype: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if self.model_name in ("llama2", "llama3"):
            dedicated_pad = "<|finetune_right_pad_id|>"
            if dedicated_pad in tokenizer.get_vocab():
                tokenizer.pad_token = dedicated_pad
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        kwargs = {"trust_remote_code": True, "torch_dtype": getattr(torch, dtype)}
        if device.startswith("cuda"):
            kwargs["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        if not device.startswith("cuda"):
            model = model.to(device)
        model.eval()
        return model, tokenizer

    def sample_training(self, seed: int):
        selected = {"benign": [], "malicious": []}
        sources = []
        for role in ("benign", "malicious"):
            for path, count in self.training_profile[role]:
                rows = self.training_line_reader(path)
                rng = random.Random(stable_seed(seed, f"train:{role}:{path.name}"))
                indices = rng.sample(range(len(rows)), count)
                selected[role].extend(rows[index] for index in indices)
                sources.append({
                    "role": role, "path": str(path.resolve()), "sha256": sha256_file(path),
                    "available": len(rows), "selected": count, "source_indices": indices,
                })
        return selected["benign"], selected["malicious"], sources

    def training_manifest(self, seed: int, sources, dtype: str):
        manifest = {
            "seed": seed, "feature_endpoint": self.training_endpoint,
            "dtype": dtype, "training_sources": sources,
            "model_config_sha256": sha256_file(self.model_path / "config.json"),
            "tokenizer_config_sha256": sha256_file(self.model_path / "tokenizer_config.json"),
        }
        if self.training_line_reader is not read_lines:
            manifest["training_text_protocol"] = self.training_line_reader.__name__
        return manifest

    def test_manifest(self, dtype: str):
        paths = [self.benign_test] + self.attack_files
        return {
            "feature_endpoint": self.test_endpoint,
            "dtype": dtype,
            "files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths],
            "model_config_sha256": sha256_file(self.model_path / "config.json"),
            "tokenizer_config_sha256": sha256_file(self.model_path / "tokenizer_config.json"),
        }

    def prepare_test_features(self, device, dtype, batch_size, force, model, tokenizer, extract):
        output = self.cache_root / "cache/test_features.pt"
        manifest = self.test_manifest(dtype)
        if output.is_file() and not force:
            saved = torch.load(output, map_location="cpu", weights_only=False)
            if saved.get("manifest") == manifest:
                return output
        datasets = {}
        all_prompts = []
        for path in [self.benign_test] + self.attack_files:
            records = load_records(path)
            start = len(all_prompts)
            all_prompts.extend(row["prompt"] for row in records)
            datasets[method_name(path)] = {
                "path": str(path.resolve()),
                "start": start,
                "count": len(records),
                "source_indices": [row["source_index"] for row in records],
            }
        features = extract(model, tokenizer, all_prompts, batch_size, device)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"manifest": manifest, "datasets": datasets, "features": features}, output)
        return output

    def prepare_training_features(self, seed, device, dtype, batch_size, force, model, tokenizer, extract):
        output = self.cache_root / "cache/training" / f"seed-{seed}.pt"
        benign, malicious, sources = self.sample_training(seed)
        manifest = self.training_manifest(seed, sources, dtype)
        if output.is_file() and not force:
            saved = torch.load(output, map_location="cpu", weights_only=False)
            if saved.get("manifest") == manifest:
                return output
        features = extract(model, tokenizer, benign + malicious, batch_size, device)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "manifest": manifest,
            "features": features,
            "labels": torch.tensor([0] * len(benign) + [1] * len(malicious)),
        }, output)
        return output

    def selected_test(self, seed: int, saved):
        feature_parts = []
        rows = []
        ordered = [method_name(self.benign_test)] + [method_name(path) for path in self.attack_files]
        for dataset in ordered:
            info = saved["datasets"][dataset]
            records = [
                {"local_index": index, "source_index": source}
                for index, source in enumerate(info["source_indices"])
            ]
            chosen = sample_records(records, seed, dataset, 500)
            indices = torch.tensor([info["start"] + row["local_index"] for row in chosen])
            feature_parts.append(saved["features"][indices])
            role = "benign" if dataset == method_name(self.benign_test) else "attack"
            rows.extend(
                {"dataset": dataset, "role": role, "source_index": row["source_index"]}
                for row in chosen
            )
        return torch.cat(feature_parts), rows

    def rank_cache_path(self, seed: int, k: int):
        name = f"seed-{seed}.pt" if k == 10 else f"seed-{seed}_k-{k}.pt"
        return self.cache_root / "cache/ranks" / name

    def load_rank_cache(self, seed: int, dtype: str, k: int):
        path = self.rank_cache_path(seed, k)
        if not path.is_file():
            return None
        saved = torch.load(path, map_location="cpu", weights_only=False)
        try:
            self.validate_rank(saved, seed, dtype, k)
        except ValueError:
            return None
        return saved

    def prepare_ranks(self, seed: int, device: str, k: int):
        output = self.rank_cache_path(seed, k)
        train_path = self.cache_root / "cache/training" / f"seed-{seed}.pt"
        test_path = self.cache_root / "cache/test_features.pt"
        train = torch.load(train_path, map_location="cpu", weights_only=False)
        test = torch.load(test_path, map_location="cpu", weights_only=False)
        selected_features, rows = self.selected_test(seed, test)
        k_values = tuple(sorted(set((self.rank_k_values or (k,)) + (k,))))
        manifest = {
            "seed": seed,
            "k_values": list(k_values),
            "training_manifest": train["manifest"],
            "test_manifest": test["manifest"],
            "sample_rows": rows,
        }
        train_ranks = rank_features(
            train["features"], train["features"], train["labels"], k_values,
            device, exclude_self=True,
        )
        test_ranks = rank_features(
            selected_features, train["features"], train["labels"], k_values,
            device, exclude_self=False,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "manifest": manifest,
            "labels": train["labels"],
            "train_ranks": train_ranks,
            "test_ranks": test_ranks,
        }, output)
        return output

    def validate_rank(self, saved, seed: int, dtype: str, k: int):
        manifest = saved["manifest"]
        training = manifest["training_manifest"]
        testing = manifest["test_manifest"]
        if manifest["seed"] != seed or training["seed"] != seed:
            raise ValueError("Rank cache seed mismatch")
        if training["dtype"] != dtype or testing != self.test_manifest(dtype):
            raise ValueError("Rank cache feature protocol mismatch")
        if training["feature_endpoint"] != self.training_endpoint:
            raise ValueError("Rank cache training endpoint mismatch")
        _, _, sources = self.sample_training(seed)
        if training["training_sources"] != sources:
            raise ValueError("Rank cache training sources mismatch")
        expected = []
        for path in [self.benign_test] + self.attack_files:
            dataset = method_name(path)
            role = "benign" if path == self.benign_test else "attack"
            rows = sample_records(load_records(path), seed, dataset)
            expected.extend(
                {"dataset": dataset, "role": role, "source_index": row["source_index"]}
                for row in rows
            )
        if manifest["sample_rows"] != expected:
            raise ValueError("Rank cache test selection mismatch")
        labels = saved["labels"]
        if labels.shape != (1600,) or int((labels == 0).sum()) != 800 or int((labels == 1).sum()) != 800:
            raise ValueError("Rank cache training labels mismatch")
        expected_k_values = sorted(set((self.rank_k_values or (k,)) + (k,)))
        if manifest["k_values"] != expected_k_values:
            raise ValueError("Rank cache k-value protocol mismatch")
        if k not in saved["train_ranks"] or k not in saved["test_ranks"]:
            raise ValueError("Rank cache does not contain the selected k")
        train = saved["train_ranks"][k]
        test = saved["test_ranks"][k]
        if train.shape != (1600, 32) or test.shape != (len(expected), 32):
            raise ValueError("Rank cache tensor shape mismatch")
        if not torch.isfinite(train).all() or not torch.isfinite(test).all():
            raise ValueError("Rank cache contains non-finite values")
