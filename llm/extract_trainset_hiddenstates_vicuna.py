from __future__ import annotations

import torch


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
    "{{ 'ASSISTANT' }}"
    "{% endif %}"
)


def configure_tokenizer(tokenizer) -> None:
    tokenizer.padding_side = "left"
    tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    if not tokenizer.add_bos_token or tokenizer.add_eos_token:
        raise ValueError("Vicuna requires add_bos_token=True and add_eos_token=False")


def render_prompts(tokenizer, prompts: list[str]) -> list[str]:
    rendered = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ) for prompt in prompts]
    if any(not text.endswith(" ASSISTANT") or text.startswith(tokenizer.bos_token) for text in rendered):
        raise ValueError("Expected Vicuna template ending in ASSISTANT without a trailing colon")
    return rendered


def extract_endpoint_activations(model, tokenizer, prompts, batch_size: int, device: str):
    chunks = []
    max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
    for start in range(0, len(prompts), batch_size):
        rendered = render_prompts(tokenizer, prompts[start:start + batch_size])
        encoded = tokenizer(
            rendered, padding=True, truncation=True, max_length=max_length,
            add_special_tokens=True, return_tensors="pt",
        )
        for row in range(len(rendered)):
            first = int(encoded.attention_mask[row].long().argmax())
            if int(encoded.input_ids[row, first]) != tokenizer.bos_token_id:
                raise ValueError("Missing leading BOS")
            if int(encoded.input_ids[row, first + 1]) == tokenizer.bos_token_id:
                raise ValueError("Duplicate leading BOS")
            tail = [tokenizer.decode([int(token)]) for token in encoded.input_ids[row, -3:]]
            if not bool(encoded.attention_mask[row, -1]) or tail != ["SS", "IST", "ANT"]:
                raise ValueError(f"Expected final ANT token at -1, got {tail}")
        with torch.inference_mode():
            outputs = model.model(
                input_ids=encoded.input_ids.to(device),
                attention_mask=encoded.attention_mask.to(device),
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
        states = torch.stack([layer[:, -1, :] for layer in outputs.hidden_states[1:]], dim=1)
        chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
        print(f"ANT (-1) features {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
        del outputs, states, encoded
    return torch.cat(chunks, dim=0)


def extract_trainset_hiddenstates(device, tokenizer, model, benign_prompts, malicious_prompts, batch_size: int):
    features = extract_endpoint_activations(
        model, tokenizer, benign_prompts + malicious_prompts, batch_size, device,
    )
    labels = torch.tensor([0] * len(benign_prompts) + [1] * len(malicious_prompts))
    return features, labels
