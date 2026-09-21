import os
from JailbreakDetector_llama3 import JailbreakDetector
import torch
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import json
from runtime import LLM_DIR, configure, label_from_path, parse_args
from Indicator_analysis_drawing import *
from extract_AC_json import extract_accuracy_to_excel
from extract_trainset_hiddenstates_llama3 import extract_trainset_hiddenstates
from draw_auroc import evaluate_attack_auroc

try:
    sys.path.append(str(LLM_DIR))
    from utils.string_utils import load_conversation_template, autodan_SuffixManager
except ImportError as e:
    sys.exit(1)

def list_available_attacks(attack_dir):
    if not os.path.isdir(attack_dir):
        return []
    files = [f for f in os.listdir(attack_dir) if f.lower().endswith('.json')]
    return [os.path.splitext(f)[0] for f in sorted(files)]


def load_prompts_from_attack_json(file_path: str):
    prompts = []
    true_label = label_from_path(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
            data = json.load(f)
    for item in data:
        if not isinstance(item, dict):
            continue
        prompt = item.get('jailbreak')
        if prompt and isinstance(prompt, str):
            prompts.append({"prompt": prompt, "true_label": true_label})
    return prompts


def get_train_dataset(benign_path_list, malicious_path_list):
    benign_prompts = []
    malicious_prompts = []
    for b_path in benign_path_list:
        with open(b_path[0], "r", errors='ignore') as b_f:
            b_f_content = b_f.readlines()
            benign_prompts.extend(random.sample(b_f_content, min(b_path[1], len(b_f_content))))
    for m_path in malicious_path_list:
        with open(m_path[0], "r") as m_f:
            m_f_content = m_f.readlines()
            malicious_prompts.extend(random.sample(m_f_content, min(m_path[1], len(m_f_content))))
    return benign_prompts, malicious_prompts


def predict(prompt_text):
    pred_label_str, pred_label, anomaly_score = detector.predict(prompt_text=prompt_text, return_score=True)
    return pred_label_str, pred_label, anomaly_score


def eval(attack_file_path_list):
    def get_last_two_levels(path):
        normalized_path = os.path.normpath(path)
        path_parts = normalized_path.split(os.sep)
        last_two_parts = path_parts[-2:] if len(path_parts) >= 2 else path_parts
        dir_name = last_two_parts[0]
        file_name = last_two_parts[1] if len(last_two_parts) > 1 else ""
        file_name_without_ext = os.path.splitext(file_name)[0]
        return f"{dir_name}_{file_name_without_ext}"

    def is_already_detected(attack_key, your_flag):
        if not os.path.exists(f"./{your_flag}/report"):
            return False
        report_file = os.path.join(f"./{your_flag}/report", f"{attack_key}_report.json")
        return os.path.exists(report_file)

    for attack_file_path in tqdm(attack_file_path_list, desc="Evaluating attack types"):
        results_detail = []
        current_attack_key = get_last_two_levels(attack_file_path)

        if is_already_detected(current_attack_key, your_flag):
            continue

        if os.path.basename(attack_file_path) == "autodan_1.json":
            attack_key = current_attack_key
            with open(attack_file_path, 'r', encoding='utf-8') as f:
                autodan_data = json.load(f)

            if not isinstance(autodan_data, list):
                continue

            conv_template = load_conversation_template(template_name)
            total = len(autodan_data)
            if total > 500:
                autodan_data = random.sample(autodan_data, 500)
            total = len(autodan_data)
            for i, item in enumerate(tqdm(autodan_data, desc="Predicting AutoDAN samples")):
                goal = (item.get('goal') or item.get('instruction') or "").strip()
                jailbreak = (item.get('jailbreak') or "").strip()
                p_suffix = jailbreak[len(goal):].strip() if len(jailbreak) >= len(goal) else jailbreak
                target = item.get('target')

                s_manager = autodan_SuffixManager(
                    tokenizer=tokenizer,
                    conv_template=conv_template,
                    instruction=goal,
                    target=target,
                    adv_string=p_suffix
                )
                input_ids = s_manager.get_input_ids(adv_string=p_suffix)

                pred_label_str, pred_label, anomaly_score = detector.predict(input_ids=input_ids, return_score=True)

                results_detail.append({
                    "Sample_Index": i + 1,
                    "prompt": jailbreak[:500] + "..." if len(jailbreak) > 500 else jailbreak,
                    "True_Label": 1,
                    "Predicted_Label": pred_label,
                    "Anomaly_Score": round(anomaly_score, 4),
                    "Prediction_Result": pred_label_str
                })

        else:
            attack_key = current_attack_key
            test_samples = load_prompts_from_attack_json(attack_file_path)
            if len(test_samples) == 0:
                continue
            total = len(test_samples)
            if len(test_samples) > 500:
                test_samples = random.sample(test_samples, 500)
            total = len(test_samples)

            for idx, sample in enumerate(tqdm(test_samples, desc="Predicting normal attack samples")):
                prompt = sample["prompt"]
                true_label = sample["true_label"]

                pred_label_str, pred_label, anomaly_score = detector.predict(prompt_text=prompt, return_score=True)

                results_detail.append({
                    "Sample_Index": idx + 1,
                    "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
                    "True_Label": true_label,
                    "Predicted_Label": pred_label,
                    "Anomaly_Score": round(anomaly_score, 4),
                    "Prediction_Result": pred_label_str
                })
        generate_report(attack_key, results_detail, your_flag, total)
    extract_accuracy_to_excel(your_flag)


if __name__ == '__main__':
    import time

    start_time = time.time()
    args = parse_args("llama3", 35, 10, 500, 512)
    configure(args)
    your_flag = f"{args.output_dir}/seed-{args.seed}_k-{args.k}_trees-{args.n_estimators}_samples-{args.max_samples}"

    benign_train_set_list = [
        ["datasets/train_data/databricks-dolly-15k.txt", 300],
        ["datasets/train_data/alpaca.txt", 300],
        ["datasets/train_data/non_refusal_prompts_with_responses_80k.txt", 200],
    ]
    malicious_train_set_list = [
        ['datasets/train_data/AdvBench.txt', 100],
        ['datasets/train_data/MaliciousInstruct.txt', 100],
        ['datasets/train_data/PKU-SafeRLHF-prompts_3-6k.txt', 600],
    ]

    template_name = 'llama-3'
    attack_dir = "datasets/llama3_test"
    attack_file_path_list = [os.path.join(attack_dir, attack_key) for attack_key in sorted(os.listdir(attack_dir)) if attack_key.endswith('.json')]

    model_path = "model/llama3"
    device = args.device
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={'': args.device},
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    benign_prompts, malicious_prompts = get_train_dataset(
        benign_train_set_list,
        malicious_train_set_list
    )
    background_layered_activations, all_labels = extract_trainset_hiddenstates(
        your_flag,
        device,
        tokenizer,
        model,
        benign_prompts,
        malicious_prompts
    )
    detector = JailbreakDetector(
        model=model,
        tokenizer=tokenizer,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        your_flag=your_flag,
        n_estimators=args.n_estimators,
        random_state=args.seed,
        max_samples=args.max_samples,
        k_nb=args.k
    )
    eval(attack_file_path_list)
    exp_dict = f"{your_flag}/report/"
    test_dict_name = "llama3_test"
    evaluate_attack_auroc(exp_dict, test_dict_name)