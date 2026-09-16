"""Shared model loading and argument helpers for example scripts."""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_attack import AdaptiveGCGConfig


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model and GCG arguments to an example CLI parser."""
    parser.add_argument(
        "--model",
        required=True,
        help="Target causal language model path or Hugging Face ID",
    )
    parser.add_argument("--prompt", required=True, help="User prompt to optimize")
    parser.add_argument("--target", required=True, help="Desired target prefix")
    parser.add_argument("--device", default="cuda", help="Model device, e.g. cuda or cpu")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Model dtype, e.g. float16 or bfloat16",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--search-width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-replace", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=0)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--sequence-weight", type=float, default=1.0)
    parser.add_argument("--last-token-offset", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-non-ascii", action="store_true")
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--no-filter-ids", action="store_true")


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype_name: str,
) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """Load the target model and ensure a padding token is available."""
    if not hasattr(torch, dtype_name):
        raise ValueError("unknown torch dtype: %s" % dtype_name)
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_config(args: argparse.Namespace) -> AdaptiveGCGConfig:
    """Convert example CLI arguments into the adaptive attack configuration."""
    return AdaptiveGCGConfig(
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        batch_size=args.batch_size,
        n_replace=args.n_replace,
        buffer_size=args.buffer_size,
        feature_weight=args.feature_weight,
        sequence_weight=args.sequence_weight,
        last_token_offset=args.last_token_offset,
        seed=args.seed,
        allow_non_ascii=args.allow_non_ascii,
        use_prefix_cache=not args.no_prefix_cache,
        filter_ids=not args.no_filter_ids,
        verbosity="INFO",
    )


def generate_text(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt: str,
    suffix: str,
    device: str,
    max_new_tokens: int = 128,
) -> str:
    """Append the searched suffix to the prompt and run greedy generation."""
    messages = [{"role": "user", "content": prompt + suffix}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = output_ids[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def print_result(
    result: object,
    generation: str,
) -> None:
    """Print a common result summary for examples."""
    print("best_string:", getattr(result, "best_string"))
    print("best_loss:", getattr(result, "best_loss"))
    print("sequence_loss:", getattr(result, "sequence_loss"))
    print("feature_loss:", getattr(result, "feature_loss"))
    print("generation:", generation)
