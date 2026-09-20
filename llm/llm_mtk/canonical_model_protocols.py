from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import canonical_best_auroc as base

HERE = Path(__file__).resolve().parent / "canonical_assets"
REPO = Path(__file__).resolve().parent.parent

VICUNA_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{% set loop_messages = messages %}"
    "{% set system_message = \"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\" %}"
    "{% endif %}"
    "{{ system_message + ' ' }}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'ASSISTANT: ' + message['content'] + eos_token + ' ' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)

SPECS = {
    "mistral": {
        "model_path": REPO / "model/mistral_7b",
        "test_dir": REPO / "datasets/mistral_test",
        "benign_file": "toxic-chat_benign_0.json",
        "training_endpoint": "mistral_slash_token_transformer_layers_1_32",
        "test_endpoint": "mistral_slash_token_embedding_plus_layers_1_31_project_compatible",
    },
    "vicuna": {
        "model_path": REPO / "model/vicuna-7b-v1_5",
        "test_dir": REPO / "datasets/vicuna_test",
        "benign_file": "toxic-chat_benign_0.json",
        "training_endpoint": "vicuna_final_ASSISTANT_colon_transformer_layers_1_32",
        "test_endpoint": "vicuna_final_ASSISTANT_colon_transformer_layers_1_32",
    },
}


def configure(model_name: str):
    spec = SPECS[model_name]
    base.HERE = HERE / model_name
    base.MODEL_PATH = spec["model_path"]
    base.TEST_DIR = spec["test_dir"]
    base.BENIGN_TEST = spec["test_dir"] / spec["benign_file"]

    def files():
        return [
            path for path in sorted(base.TEST_DIR.glob("*.json"))
            if "toxic-chat" not in path.name.lower() and "normal_" not in path.name.lower()
        ]

    base.attack_files = files
    return spec


def configure_tokenizer(tokenizer, model_name: str):
    tokenizer.padding_side = "left"
    if model_name == "vicuna" and tokenizer.chat_template is None:
        tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        if model_name == "vicuna" and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token


def load_model(model_name: str, device: str, dtype: str):
    spec = SPECS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(spec["model_path"], trust_remote_code=True)
    configure_tokenizer(tokenizer, model_name)
    kwargs = {"trust_remote_code": True, "torch_dtype": getattr(torch, dtype)}
    if device.startswith("cuda"):
        kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(spec["model_path"], **kwargs)
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return model, tokenizer


def render(tokenizer, prompts):
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    ) for prompt in prompts]


def endpoint_indices(model_name, tokenizer, input_ids, attention_mask):
    if model_name == "vicuna":
        return torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1,
            dtype=torch.long, device=input_ids.device,
        )

    targets = []
    for row in range(input_ids.shape[0]):
        valid = torch.nonzero(attention_mask[row], as_tuple=True)[0]
        target = int(valid[-1])
        token = tokenizer.decode([int(input_ids[row, target])])
        while target > 0 and ("INST" in token or "]" in token):
            target -= 1
            token = tokenizer.decode([int(input_ids[row, target])])
        targets.append(target)
    return torch.tensor(targets, dtype=torch.long, device=input_ids.device)


def extractor(model_name: str, phase: str):
    def extract(model, tokenizer, prompts, batch_size, device):
        chunks = []
        max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
        rendered = render(tokenizer, prompts)
        for start in range(0, len(rendered), batch_size):
            batch = rendered[start:start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            targets = endpoint_indices(model_name, tokenizer, input_ids, attention_mask)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            row_ids = torch.arange(input_ids.shape[0], device=device)
            if model_name == "mistral" and phase == "test":
                layers = outputs.hidden_states[:-1]
            else:
                layers = outputs.hidden_states[1:]
            states = torch.stack([layer[row_ids, targets, :] for layer in layers], dim=1)
            chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
            print(
                f"{model_name} {phase} features "
                f"{min(start + batch_size, len(rendered))}/{len(rendered)}",
                flush=True,
            )
        return torch.cat(chunks, dim=0)

    return extract


def prepare(args):
    spec = configure(args.model)
    model, tokenizer = load_model(args.model, args.device, args.dtype)

    base.FEATURE_ENDPOINT = spec["test_endpoint"]
    base.extract_activations = extractor(args.model, "test")
    base.prepare_test_features(
        args.device, args.dtype, args.batch_size, args.force,
        model=model, tokenizer=tokenizer,
    )

    for seed in args.seeds:
        rank_path = base.HERE / "cache/ranks" / f"seed-{seed}.pt"
        if rank_path.exists() and not args.force:
            print(f"{args.model} seed {seed}: existing rank cache skipped", flush=True)
            continue
        base.FEATURE_ENDPOINT = spec["training_endpoint"]
        base.extract_activations = extractor(args.model, "training")
        training_path = base.prepare_training_features(
            seed, args.device, args.dtype, args.batch_size, args.force,
            model=model, tokenizer=tokenizer,
        )
        base.prepare_ranks(seed, args.device, args.force)
        training_path.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize(model_name: str):
    configure(model_name)
    for case in ("seed_only", "hyperparameter_search"):
        best_path = base.HERE / case / "best/best_result.json"
        if not best_path.exists():
            continue
        best = json.loads(best_path.read_text(encoding="utf-8"))
        rank = torch.load(
            base.HERE / "cache/ranks" / f"seed-{best['seed']}.pt",
            map_location="cpu", weights_only=False,
        )
        manifest = {
            "result": best,
            "selection_metric": "unweighted macro mean of per-attack AUROCs",
            "score_source": "raw unrounded negative Isolation Forest decision_function",
            "training_reference_counts": {"benign": 800, "malicious": 800},
            "test_sampling": "500 benign and min(500, all valid) per jailbreak dataset",
            "autodan_input": "jailbreak prompt only; target answer excluded",
            "rank_manifest": rank["manifest"],
        }
        base.write_json(base.HERE / case / "best/run_manifest.json", manifest)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("command", choices=["prepare", "seed-search", "hyper-search", "finalize"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure(args.model)
    base.set_determinism(42)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "seed-search":
        base.command_seed_search(args)
    elif args.command == "hyper-search":
        base.command_hyper_search(args)
    else:
        finalize(args.model)


if __name__ == "__main__":
    main()
