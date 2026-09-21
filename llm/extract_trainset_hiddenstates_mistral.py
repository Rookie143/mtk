import torch
import os
from tqdm import tqdm

def extract_trainset_hiddenstates(your_flag, device, tokenizer, model, benign_prompts, malicious_prompts):
    load_path = f"./{your_flag}/saved_features_and_labels.pt"
    need_extract = True

    if os.path.exists(load_path):
        try:
            loaded_data = torch.load(load_path, map_location=device)
            background_layered_activations = loaded_data["background_layered_activations"]
            all_labels = loaded_data["labels"]
            loaded_layers = background_layered_activations.shape[1]
            current_model_layers = model.config.num_hidden_layers
            if loaded_layers != current_model_layers:
                need_extract = True
            else:
                need_extract = False
        except Exception as e:
            need_extract = True

    if need_extract:
        all_activations = []

        def process_batch(prompts, desc_text):
            batch_acts = []
            for sentence in tqdm(prompts, desc=desc_text):
                messages = [{"role": "user", "content": sentence}]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False
                ).to(model.device)

                if tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(model.device)
                else:
                    attention_mask = torch.ones_like(input_ids).to(model.device)

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    hidden_states = outputs.hidden_states[1:]

                    seq_len = input_ids.shape[1]

                    if tokenizer.padding_side == 'right':
                        last_idx_tensor = attention_mask.sum(dim=1) - 1
                    else:
                        last_idx_tensor = input_ids.new_full((input_ids.shape[0],), seq_len - 1)

                    scalar_idx = last_idx_tensor[0].item()

                    curr_token_str = tokenizer.decode([input_ids[0, scalar_idx].item()])
                    while scalar_idx > 0 and ("INST" in curr_token_str or "]" in curr_token_str):
                        scalar_idx -= 1
                        curr_token_str = tokenizer.decode([input_ids[0, scalar_idx].item()])

                    activations = [layer_hidden_state[0, scalar_idx, :].cpu().clone() for layer_hidden_state in
                                   hidden_states]

                    del input_ids, outputs, hidden_states
                    torch.cuda.empty_cache()

                batch_acts.append(activations)
            return batch_acts

        benign_acts = process_batch(benign_prompts, "Extracting benign sample features")
        all_activations.extend(benign_acts)

        malicious_acts = process_batch(malicious_prompts, "Extracting malicious sample features")
        all_activations.extend(malicious_acts)

        benign_labels = torch.zeros(len(benign_prompts), device=device)
        malicious_labels = torch.ones(len(malicious_prompts), device=device)
        all_labels = torch.cat([benign_labels, malicious_labels], dim=0)

        num_layers = len(all_activations[0])

        layered_activations = []
        for l in range(num_layers):
            layer_feats = torch.stack([sample_feats[l] for sample_feats in all_activations], dim=0)
            layered_activations.append(layer_feats)

        background_layered_activations = torch.stack(layered_activations, dim=1).to(device)

        save_dir = os.path.dirname(load_path)
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "background_layered_activations": background_layered_activations,
            "labels": all_labels
        }, load_path)

    return background_layered_activations, all_labels