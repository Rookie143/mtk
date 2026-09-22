from __future__ import annotations

import torch

from feature_protocol import render_prompts


def extract(model, tokenizer, prompts, batch_size, device):
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
        captured = []
        handles = []

        def capture(module, inputs, output):
            states = output[0] if isinstance(output, tuple) else output
            captured.append(states[:, -1, :].detach().clone())

        for layer in model.model.layers[:31]:
            handles.append(layer.register_forward_hook(capture))
        handles.append(model.model.norm.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                model.model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False, use_cache=False, return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 32:
            raise RuntimeError(f"Captured {len(captured)} endpoints instead of 32")
        chunks.append(torch.stack(captured, dim=1).to(device="cpu", dtype=torch.float16))
        print(f"features {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return torch.cat(chunks, dim=0)
