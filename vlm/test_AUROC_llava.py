import torch
import csv
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import numpy as np
from sklearn.metrics import precision_recall_curve,auc,precision_score,roc_curve
from tqdm import tqdm
from JailbreakDetector_llava import JailbreakDetector
from transformers import AutoProcessor, AutoModelForVision2Seq
from load_datasets import *

def find_threshold_for_target_fpr(scores_0, target_fpr=0.05):
    sorted_scores = np.sort(scores_0)[::-1]
    n_positive = int(np.ceil(len(sorted_scores) * target_fpr))
    if n_positive == 0:
        threshold = sorted_scores[0] + 1e-6
    elif n_positive >= len(sorted_scores):
        threshold = sorted_scores[-1] - 1e-6
    else:
        threshold = sorted_scores[n_positive - 1]
    
    actual_fpr = (scores_0 >= threshold).sum() / len(scores_0)
    print(f"FPR=0.05,threshold={threshold}")
    return threshold

def calculate_metrics_with_threshold(scores, labels, threshold):
    pred_labels = (scores >= threshold).astype(int)
    n_negative = (labels == 0).sum()
    false_positive = ((pred_labels == 1) & (labels == 0)).sum()
    fpr = false_positive / n_negative if n_negative > 0 else 0.0
    precision = precision_score(labels, pred_labels, zero_division=0.0)
    print(f"FPR: {fpr:.4f}")
    print(f"Precision: {precision:.4f}")
    return fpr, precision

def evaluate_AUPRC(true_labels, scores):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(true_labels, scores)
    auprc = auc(recall_arr, precision_arr)
    return auprc

def evaluate_AUROC(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    auroc = auc(fpr, tpr)
    return auroc   
   
def main(flag):
    benign_train_data = load_vqa_dataset_for_train()+load_usb_datasset_for_train()
    malicious_train_data = load_sd_advbench_for_train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf",device_map='auto',torch_dtype=torch.float16)
    all_activations = []  
    all_labels = []   
    if os.path.exists(f"./experimental_results/{flag}/saved_features_and_labels.pt"):
        load_path = f"./experimental_results/{flag}/saved_features_and_labels.pt"
        loaded_data = torch.load(load_path, map_location=device)
        background_layered_activations = loaded_data["background_layered_activations"]
        all_labels = loaded_data["labels"]
    else:
        benign_prompts = benign_train_data

        all_activations = []
        for sentence in tqdm(benign_prompts, desc="Extract features of benign samples"):
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": sentence[1]},
                            {"type": "text", "text": sentence[0]}
                        ]
                    },
                ]
            with torch.no_grad(): 
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1:]
                activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
                del inputs, outputs, hidden_states
                torch.cuda.empty_cache()
            all_activations.append(activations)
            benign_labels = torch.zeros(len(benign_prompts), device=device)
        malicious_prompts = malicious_train_data
        for sentence in tqdm(malicious_prompts, desc="Extract features of malicious samples"):
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": sentence[1]},
                            {"type": "text", "text": sentence[0]}
                        ]
                    },
                ]
            with torch.no_grad(): 
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1:]
                activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
                del inputs, outputs, hidden_states
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
        if not os.path.exists(f"./experimental_results/{flag}"):
            os.mkdir(f"./experimental_results/{flag}")
        save_path = f"./experimental_results/{flag}/saved_features_and_labels.pt"
        torch.save({
            "background_layered_activations": background_layered_activations,
            "labels": all_labels
        }, save_path)

    detector = JailbreakDetector(
        model=model,
        processor=processor,
        background_layered_activations=background_layered_activations,
        all_labels=all_labels,
        flag = flag,
        n_estimators=100,
        random_state=42,
        k_nb=10,
    )
    datasets = {}     
    results = {}   
    datasets["MM-SafetyBench+MM-Vet"] = load_mm_safety_bench_all() + load_mm_vet_v2()
    # datasets["MM-SafetyBench+MM-vqa"] = load_mm_safety_bench_all() + load_vqa()
    datasets["FigImg+MM-Vet"] = load_FigImg() + load_mm_vet_v2()
    datasets["JBV28K_JBtxt_SDimg+MM-Vet"] = load_JailBreakV_JBtxt_SDimg() + load_mm_vet_v2()
    datasets["MM-SafetyBench+usb"] = load_mm_safety_bench_all() +load_usb_datasset() 
    datasets["MM-Vet_all"] = load_mm_vet_v2(is_all=True)

    total_datasets = len(datasets)    
    print(f"Starting evaluation of {total_datasets} datasets...")
    
    for idx, (dataset_name, dataset) in enumerate(datasets.items(), 1):
        print(f"Processing dataset {idx}/{total_datasets}: {dataset_name}")
        true_labels = []
        scores = []
        for i in tqdm(dataset, desc="Processing Dataset", leave=True, ncols=100):
            _, score = detector.predict([i['txt'], i['img']])
            true_labels.append(i["toxicity"])
            scores.append(score)
        dataset_data = [
            [i["txt"] for i in dataset],
            [i["img"] for i in dataset],
            [i["toxicity"] for i in dataset],
            scores
        ]
        transposed_data = list(zip(*dataset_data))
        transposed_data = [list(row) for row in zip(*dataset_data)]
        if not os.path.exists(f"./experimental_results/{flag}/results"):
            os.mkdir(f"./experimental_results/{flag}/results")
        with open(f"./experimental_results/{flag}/results/test_llava_{dataset_name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(['txt', 'img', 'true_labels', 'score'])
            writer.writerows(transposed_data)
        AUPRC = evaluate_AUPRC(true_labels, scores)
        AUROC = evaluate_AUROC(true_labels, scores)    
        if dataset_name != "MM-Vet_all":        
            results[dataset_name] = (AUPRC,AUROC)
            print(f"AUPRC for {dataset_name}: {AUPRC}")
            print(f"AUROC for {dataset_name}: {AUROC}")
        with open(f"./experimental_results/{flag}/results/llava_AUROC_result.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dataset Name", "AUPRC","AUROC"])  
            for dataset_name, result in results.items():
                if result is not None:  
                    writer.writerow([dataset_name, f"{result[0]:.4f}", f"{result[1]:.4f}"])
    scores_0_only_csv_file_path = f"./experimental_results/{flag}/results/test_llava_MM-Vet_all.csv"  # 替换为你的CSV文件路径（如：/Users/xxx/data.csv 或 D:/xxx/data.csv）
    to_test_csv_file_path = f"./experimental_results/{flag}/results/test_llava_MM-SafetyBench+usb.csv"
    scores_0_df = pd.read_csv(scores_0_only_csv_file_path)
    to_test_df = pd.read_csv(to_test_csv_file_path)
    required_cols = ["true_labels", "score"]
    missing_cols = [col for col in required_cols if col not in scores_0_df.columns]
    scores_0_only = scores_0_df[scores_0_df["true_labels"] == 0]["score"].values 
    threshold = find_threshold_for_target_fpr(scores_0_only, target_fpr=0.05)
    all_scores = to_test_df["score"].values
    all_labels = to_test_df["true_labels"].values
    fpr, precision = calculate_metrics_with_threshold(all_scores, all_labels, threshold)
if __name__ == '__main__':
    import datetime
    if len(sys.argv) > 1:
        arg_value = sys.argv[1]
    else:
        arg_value = datetime.now().strftime("%Y%m%d_%H%M%S")
    main(arg_value)












        

    
    
    
    
    
    











    
