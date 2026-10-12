from __future__ import annotations

import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_determinism(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def read_lines(path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as stream:
        return [line for line in stream.readlines() if line.strip()]


def render_prompts(tokenizer, prompts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def rank_features(
    queries,
    background,
    labels,
    k_values,
    device,
    exclude_self=False,
    batch_size=64,
):
    n_queries, n_layers, _ = queries.shape
    output = {k: torch.empty((n_queries, n_layers), dtype=torch.float32) for k in k_values}
    background = background.to(device=device)
    labels = labels.to(device)
    benign_count = int((labels == 0).sum()) - int(exclude_self)
    if max(k_values) > benign_count:
        raise ValueError(f"k={max(k_values)} exceeds available benign references {benign_count}")
    positions = torch.arange(1, len(labels) + 1, device=device, dtype=torch.float32)
    for layer in range(n_layers):
        references = background[:, layer, :]
        for start in range(0, n_queries, batch_size):
            stop = min(start + batch_size, n_queries)
            query = queries[start:stop, layer, :].to(device=device, dtype=background.dtype)
            distances = (references.unsqueeze(0) - query.unsqueeze(1)).norm(p=2, dim=2)
            if exclude_self:
                local = torch.arange(stop - start, device=device)
                distances[local, torch.arange(start, stop, device=device)] = torch.inf
            order = distances.argsort(dim=1)
            benign_sorted = labels[order].eq(0)
            ranked_positions = positions.expand(stop - start, -1).masked_fill(
                ~benign_sorted, torch.inf
            )
            nearest_positions = ranked_positions.topk(
                max(k_values), largest=False, dim=1
            ).values
            cumulative = nearest_positions.cumsum(dim=1)
            for k in k_values:
                output[k][start:stop, layer] = (cumulative[:, k - 1] / k).cpu()
    return output


class FeatureProtocol:
    def __init__(self, model_path):
        self.model_path = model_path

    def model_and_tokenizer(self, device, dtype):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        dedicated_pad = "<|finetune_right_pad_id|>"
        if dedicated_pad in tokenizer.get_vocab():
            tokenizer.pad_token = dedicated_pad
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        options = {
            "trust_remote_code": True,
            "torch_dtype": getattr(torch, dtype),
        }
        if device.startswith("cuda"):
            options["device_map"] = {"": device}
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **options)
        if not device.startswith("cuda"):
            model = model.to(device)
        model.eval()
        return model, tokenizer
