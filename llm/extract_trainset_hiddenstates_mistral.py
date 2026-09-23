from contextlib import redirect_stderr, redirect_stdout
import os

import torch

from feature_protocol import FeatureProtocol, render_prompts, set_determinism


ATTACK_FILES = ["JailJudge_all_1.json", "ijp_0.json", "nonagcg_1.json", "autodan_1.json", "drattack_1.json", "pair_1.json", "pap_gpt3.5_1.json", "pap_gpt4_1.json", "pap_llama2_1.json", "saa_1.json", "tap_1.json", "zulu_1.json"]
TRAINING_ENDPOINT = "mistral_slash_token_transformer_layers_1_32"
TEST_ENDPOINT = "mistral_slash_token_embedding_plus_layers_1_31_project_compatible"
CANONICAL_RANK_K_VALUES = (1, 3, 5, 10, 15, 20, 25, 30)


def read_mistral_training_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as stream:
        rows = [line for line in stream.readlines() if line.strip()]
    if path.name != "alpaca.txt":
        rows = [line.replace("\r\n", "\n") for line in rows]
    return rows


def endpoint_indices(tokenizer, input_ids, attention_mask):
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


def mistral_extract(phase):
    def extract(model, tokenizer, prompts, batch_size, device):
        chunks = []
        max_length = min(int(getattr(model.config, "max_position_embeddings", 4096)), 4096)
        rendered = render_prompts(tokenizer, prompts)
        for start in range(0, len(rendered), batch_size):
            encoded = tokenizer(
                rendered[start:start + batch_size], padding=True, truncation=True,
                max_length=max_length, add_special_tokens=False, return_tensors="pt",
            )
            input_ids = encoded.input_ids.to(device)
            attention_mask = encoded.attention_mask.to(device)
            targets = endpoint_indices(tokenizer, input_ids, attention_mask)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True,
                )
            row_ids = torch.arange(input_ids.shape[0], device=device)
            layers = outputs.hidden_states[:-1] if phase == "test" else outputs.hidden_states[1:]
            states = torch.stack([layer[row_ids, targets, :] for layer in layers], dim=1)
            chunks.append(states.detach().to(device="cpu", dtype=torch.float16))
        return torch.cat(chunks, dim=0)
    return extract


def extract_trainset_hiddenstates(
    seed, device, dtype, batch_size, benign_train_set_list, malicious_train_set_list,
    rebuild_cache=False, force_rebuild_cache=False, k=10,
):
    protocol = FeatureProtocol(
        "mistral", "mistral_7b", ATTACK_FILES,
        benign_train_set_list, malicious_train_set_list,
        TRAINING_ENDPOINT, TEST_ENDPOINT,
        training_line_reader=read_mistral_training_lines,
        rank_k_values=CANONICAL_RANK_K_VALUES,
    )
    set_determinism(42)
    rank_path = protocol.rank_cache_path(seed, k)
    saved = None
    if not force_rebuild_cache and not rebuild_cache:
        saved = protocol.load_rank_cache(seed, dtype, k)
        if saved is not None:
            return saved
    if saved is None:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                model, tokenizer = protocol.model_and_tokenizer(device, dtype)
                protocol.prepare_test_features(device, dtype, batch_size, force_rebuild_cache, model, tokenizer, mistral_extract("test"))
                training_path = protocol.prepare_training_features(seed, device, dtype, batch_size, force_rebuild_cache, model, tokenizer, mistral_extract("training"))
                protocol.prepare_ranks(seed, device, k)
                training_path.unlink(missing_ok=True)
        del model, tokenizer
    saved = torch.load(rank_path, map_location="cpu", weights_only=False)
    protocol.validate_rank(saved, seed, dtype, k)
    return saved
