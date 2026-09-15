import torch
import os
from tqdm import tqdm


def extract_trainset_hiddenstates(your_flag, device, tokenizer, model, benign_prompts, malicious_prompts):
    all_activations = []
    all_labels = []
    if os.path.exists(f"./{your_flag}/saved_features_and_labels.pt"):
        load_path = f"./{your_flag}/saved_features_and_labels.pt"
        loaded_data = torch.load(load_path, map_location=device)
        background_layered_activations = loaded_data["background_layered_activations"]
        all_labels = loaded_data["labels"]
    else:
        total_benign = len(benign_prompts)
        all_activations = []
        for sentence in tqdm(benign_prompts, desc="Extracting benign sample features"):
            messages = [{"role": "user", "content": sentence}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False
            ).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(
                device) if tokenizer.pad_token_id else torch.ones_like(input_ids).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[1:]
                activations = [layer_hidden_state[0, -1, :] for layer_hidden_state in hidden_states]
            all_activations.append(activations)
            benign_labels = torch.zeros(len(benign_prompts), device=device)

        for sentence in tqdm(malicious_prompts, desc="Extracting malicious sample features"):
            messages = [{"role": "user", "content": sentence}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False
            ).to(device)
            attention_mask = (input_ids != tokenizer.pad_token_id).long().to(
                device) if tokenizer.pad_token_id else torch.ones_like(input_ids).to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[1:]
                activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
                del input_ids, outputs, hidden_states
                torch.cuda.empty_cache()
            all_activations.append(activations)
            malicious_labels = torch.ones(len(malicious_prompts), device=device)
            all_labels = torch.cat([benign_labels, malicious_labels], dim=0)
        num_layers = len(all_activations[0])
        layered_activations = []
        for l in range(num_layers):
            layer_feats = torch.stack([sample_feats[l] for sample_feats in all_activations], dim=0)
            layered_activations.append(layer_feats)
        background_layered_activations = torch.stack(layered_activations, dim=1)
        save_path = f"./{your_flag}/saved_features_and_labels.pt"
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "background_layered_activations": background_layered_activations,
            "labels": all_labels
        }, save_path)
    return background_layered_activations, all_labels