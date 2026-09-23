from __future__ import annotations

from pathlib import Path

import torch


VICUNA_FUSION_CHAT_TEMPLATE = (
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


def configure_tokenizer(tokenizer) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.chat_template is None:
        tokenizer.chat_template = VICUNA_FUSION_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token


def render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ) for prompt in prompts]


def extract_endpoint_activations(model, tokenizer, prompts, batch_size: int, device: str, endpoint: str):
    if endpoint not in {"colon", "ist"}:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    chunks = []
    max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    rendered = render_prompts(tokenizer, prompts)
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
        offset = -1 if endpoint == "colon" else -3
        targets = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] + offset, dtype=torch.long, device=device
        )
        if endpoint == "ist":
            decoded = [tokenizer.decode([int(input_ids[row, targets[row]])]) for row in range(input_ids.shape[0])]
            if any(piece != "IST" for piece in decoded):
                raise ValueError(f"Vicuna IST endpoint invariant failed: {decoded}")
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        row_ids = torch.arange(input_ids.shape[0], device=device)
        layers = outputs.hidden_states[1:model.config.num_hidden_layers + 1]
        states = torch.stack([layer[row_ids, targets, :] for layer in layers], dim=1)
        chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
        print(f"{endpoint} features {min(start + batch_size, len(rendered))}/{len(rendered)}", flush=True)
        del outputs, layers, states, input_ids, attention_mask, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def extract_dual_endpoint_activations(
    model, tokenizer, prompts, colon_batch_size: int, ist_batch_size: int, device: str
):
    return {
        "colon": extract_endpoint_activations(
            model, tokenizer, prompts, colon_batch_size, device, "colon"
        ),
        "ist": extract_endpoint_activations(
            model, tokenizer, prompts, ist_batch_size, device, "ist"
        ),
    }


def extract_input_ids_activations(
    model, tokenizer, input_id_sequences, batch_size: int, device: str
):
    chunks = []
    for start in range(0, len(input_id_sequences), batch_size):
        sequences = input_id_sequences[start:start + batch_size]
        batch = tokenizer.pad(
            {"input_ids": [sequence.tolist() for sequence in sequences]},
            padding=True,
            return_tensors="pt",
        )
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        layers = outputs.hidden_states[1:model.config.num_hidden_layers + 1]
        states = torch.stack([layer[:, -1, :] for layer in layers], dim=1)
        chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
        print(
            f"AutoDAN target features "
            f"{min(start + batch_size, len(input_id_sequences))}/"
            f"{len(input_id_sequences)}",
            flush=True,
        )
        del outputs, layers, states, input_ids, attention_mask, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def extract_trainset_hiddenstates(
    your_flag,
    device,
    tokenizer,
    model,
    benign_prompts,
    malicious_prompts,
    colon_batch_size: int,
    ist_batch_size: int,
):
    path = Path(your_flag) / "saved_features_and_labels.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        return saved["background_layered_activations"], saved["labels"]
    features = extract_dual_endpoint_activations(
        model,
        tokenizer,
        benign_prompts + malicious_prompts,
        colon_batch_size,
        ist_batch_size,
        device,
    )
    labels = torch.tensor([0] * len(benign_prompts) + [1] * len(malicious_prompts))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "background_layered_activations": features,
        "labels": labels,
    }, path)
    return features, labels
