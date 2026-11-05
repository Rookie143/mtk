import torch

from feature_protocol import FeatureProtocol, load_records, method_name, rank_features, render_prompts, set_determinism


ATTACK_FILES = ["JailJudge_all_1.json", "ijp_1.json", "nonagcg_1.json", "autodan_1.json", "drattack_1.json", "pair_1.json", "pap_1.json", "saa_1.json", "tap_1.json", "zulu_1.json"]
TRAINING_ENDPOINT = "mistral_trimmed_inst_slash_minus1_transformer_layers_1_32"
TEST_ENDPOINT = TRAINING_ENDPOINT


def read_mistral_training_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as stream:
        rows = [line for line in stream.readlines() if line.strip()]
    if path.name != "alpaca.txt":
        rows = [line.replace("\r\n", "\n") for line in rows]
    return rows


def mistral_extract(phase):
    def extract(model, tokenizer, prompts, batch_size, device):
        chunks = []
        max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
        tokenizer.chat_template = tokenizer.chat_template.replace("' [/INST]'", "' [/'")
        rendered = render_prompts(tokenizer, prompts)
        if any(not text.endswith(" [/") for text in rendered):
            raise ValueError("Expected Mistral template ending in [/")
        for start in range(0, len(rendered), batch_size):
            encoded = tokenizer(
                rendered[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, add_special_tokens=False, return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            if not bool(attention_mask[:, -1].all()) or any(tokenizer.decode([int(token)]) != "/" for token in input_ids[:, -1]):
                raise ValueError("Expected final / token at -1; check tokenization or truncation")
            layers = []
            def capture(module, inputs, output):
                layer = output[0] if isinstance(output, tuple) else output
                layers.append(layer[:, -1, :].detach().clone())
            handles = [layer.register_forward_hook(capture) for layer in model.model.layers[:-1]]
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
            states = torch.stack(layers, dim=1)
            chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
            if start % (batch_size * 50) == 0 or start + batch_size >= len(rendered):
                print(f"Mistral {phase} / (-1) features {min(start + batch_size, len(rendered))}/{len(rendered)}", flush=True)
        return torch.cat(chunks, dim=0)
    return extract


def extract_trainset_hiddenstates(
    seed, device, dtype, batch_size, benign_train_set_list, malicious_train_set_list,
    k=10,
):
    protocol = FeatureProtocol(
        "mistral", "mistral_7b", ATTACK_FILES,
        benign_train_set_list, malicious_train_set_list,
        TRAINING_ENDPOINT, TEST_ENDPOINT,
        training_line_reader=read_mistral_training_lines,
    )
    set_determinism(42)
    model, tokenizer = protocol.model_and_tokenizer(device, dtype)
    datasets, prompts = {}, []
    for path in [protocol.benign_test] + protocol.attack_files:
        records = load_records(path)
        datasets[method_name(path)] = {
            "start": len(prompts), "source_indices": [row["source_index"] for row in records],
        }
        prompts.extend(row["prompt"] for row in records)
    features = mistral_extract("test")(model, tokenizer, prompts, batch_size, device)
    selected, rows = protocol.selected_test(seed, {"datasets": datasets, "features": features})
    del features
    benign, malicious, sources = protocol.sample_training(seed)
    background = mistral_extract("training")(model, tokenizer, benign + malicious, batch_size, device)
    del model, tokenizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    labels = torch.tensor([0] * len(benign) + [1] * len(malicious))
    k_values = (k,)
    saved = {
        "manifest": {
            "seed": seed, "k_values": list(k_values), "sample_rows": rows,
            "training_manifest": protocol.training_manifest(seed, sources, dtype),
            "test_manifest": protocol.test_manifest(dtype),
        },
        "labels": labels,
        "train_ranks": rank_features(background, background, labels, k_values, device, True),
        "test_ranks": rank_features(selected, background, labels, k_values, device, False),
    }
    protocol.validate_rank(saved, seed, dtype, k)
    return saved
