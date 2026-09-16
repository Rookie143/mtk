"""Build an MTK hidden-state reference bank from a JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    """Parse reference-bank builder arguments."""
    parser = argparse.ArgumentParser(
        description="Build an MTK hidden-state reference bank",
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face ID")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--text-key",
        default="text",
        help="Text field name in the JSONL file; default: text",
    )
    parser.add_argument(
        "--label-key",
        default="label",
        help="Label field name in the JSONL file; default: label",
    )
    parser.add_argument(
        "--benign-label",
        type=int,
        default=1,
        help="Benign sample label; default: 1",
    )
    parser.add_argument(
        "--malicious-label",
        type=int,
        default=0,
        help="Malicious sample label; default: 0",
    )
    return parser.parse_args()


def load_records(
    path: str | Path,
    text_key: str,
    label_key: str,
) -> tuple[list[str], Tensor]:
    """Read JSONL records and validate text and label fields."""
    texts: list[str] = []
    labels: list[int] = []
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "line %d is not valid JSON" % line_number
                ) from error
            if not isinstance(record, dict):
                raise ValueError("line %d must be a JSON object" % line_number)
            text = record.get(text_key)
            label = record.get(label_key)
            if not isinstance(text, str) or not text:
                raise ValueError("line %d has an invalid text field" % line_number)
            if not isinstance(label, int):
                raise ValueError("line %d label field must be an integer" % line_number)
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError("input JSONL contains no valid samples")
    return texts, torch.tensor(labels, dtype=torch.long)


def load_model(
    model_path: str,
    device: str,
    dtype_name: str,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Load the target model and tokenizer."""
    if not hasattr(torch, dtype_name):
        raise ValueError("unknown torch dtype: %s" % dtype_name)
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_messages(
    tokenizer: transformers.PreTrainedTokenizer,
    texts: list[str],
) -> list[str]:
    """Convert plain texts into model chat-template strings when available."""
    if not tokenizer.chat_template:
        return texts
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for text in texts
    ]


@torch.no_grad()
def extract_features(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    texts: list[str],
    batch_size: int,
) -> Tensor:
    """Extract every layer's last-input-token hidden state for each sample."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    device = next(model.parameters()).device
    formatted_texts = format_messages(tokenizer, texts)
    batches: list[Tensor] = []
    for start in range(0, len(formatted_texts), batch_size):
        text_batch = formatted_texts[start : start + batch_size]
        encoded = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        outputs = model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states[1:]
        last_indices = encoded["attention_mask"].sum(dim=1) - 1
        layer_features = torch.stack(
            [
                layer_state[
                    torch.arange(layer_state.shape[0], device=device),
                    last_indices,
                ]
                for layer_state in hidden_states
            ],
            dim=1,
        )
        batches.append(layer_features.float().cpu())
    return torch.cat(batches, dim=0)


def main() -> None:
    """Read data, extract hidden states, and save the reference bank."""
    args = parse_args()
    texts, labels = load_records(
        args.input,
        text_key=args.text_key,
        label_key=args.label_key,
    )
    if not torch.all(
        (labels == args.benign_label) | (labels == args.malicious_label)
    ):
        raise ValueError(
            "all input labels must be benign-label=%d or malicious-label=%d"
            % (args.benign_label, args.malicious_label)
        )

    model, tokenizer = load_model(args.model, args.device, args.dtype)
    features = extract_features(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        batch_size=args.batch_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "background_layered_activations": features,
            "labels": labels,
        },
        output_path,
    )
    print("saved:", output_path)
    print("features:", tuple(features.shape))
    print("labels:", tuple(labels.shape))


if __name__ == "__main__":
    main()
