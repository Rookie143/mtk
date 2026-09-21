import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import json


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "Confusion_Matrix": cm.tolist(),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4)
    }
    return metrics

def save_experiment_report(metrics, results_detail, attack_name, save_dir="./experiment_reports"):
    os.makedirs(save_dir, exist_ok=True)

    report = {
        "Attack_Type": attack_name,
        "Total_Samples": len(results_detail),
        "True_Malicious_Count": sum(1 for x in results_detail if x["True_Label"] == 1),
        "True_Benign_Count": sum(1 for x in results_detail if x["True_Label"] == 0),
        "Core_Metrics": metrics,
        "Experiment_Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    json_path = os.path.join(save_dir, f"{attack_name}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(save_dir, f"{attack_name}_results_detail.csv")
    df = pd.DataFrame(results_detail)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

def generate_report(attack_key, results_detail, your_flag, total):
    y_true = [item["True_Label"] for item in results_detail]
    y_pred = [item["Predicted_Label"] for item in results_detail]

    metrics = calculate_metrics(y_true, y_pred)

    save_experiment_report(metrics, results_detail, attack_key, f"./{your_flag}/report")

    malicious_count = sum(1 for item in results_detail if item["Predicted_Label"] == 1)