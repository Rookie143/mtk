import os
import glob
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate_attack_auroc(exp_dir, test_dict_name):
    toxic_chat_file = os.path.join(exp_dir, f"{test_dict_name}_toxic-chat_benign_0_results_detail.csv")

    if not os.path.exists(toxic_chat_file):
        raise FileNotFoundError(f"Toxic-chat file not found: {toxic_chat_file}")

    df_toxic = pd.read_csv(toxic_chat_file)


    benign_df = df_toxic.copy()
    benign_df["True_Label"] = 0

    result_dict = {}

    search_pattern = os.path.join(exp_dir, f"{test_dict_name}_*_1_results_detail.csv")
    for file_path in sorted(glob.glob(search_pattern)):
        file_name = os.path.basename(file_path)
        attack_name = file_name.replace(f"{test_dict_name}_", "").replace("_1_results_detail.csv", "")

        try:
            attack_df = pd.read_csv(file_path)
            attack_df["True_Label"] = 1

            df = pd.concat([benign_df, attack_df], ignore_index=True)

            y_true = df["True_Label"].values
            y_attack = (y_true == 1).astype(int)
            y_score = -df["Anomaly_Score"].values

            auroc = roc_auc_score(y_attack, y_score)
            result_dict[attack_name] = auroc

        except Exception as error:
            raise RuntimeError(f"Failed to evaluate {file_path}") from error
    result_df = pd.DataFrame(list(result_dict.items()), columns=["Attack Method", "AUROC"])
    out_path = os.path.join(exp_dir, "all_attack_auroc_results.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    experiment_directory = "vicuna/report/"
    test_dic_name = "vicuna-7b-v1.5"
    evaluate_attack_auroc(experiment_directory, test_dic_name)
