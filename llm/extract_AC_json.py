import os
import json
import pandas as pd
import subprocess


def extract_accuracy_to_excel(your_flag='.'):
    folder_path = f"./{your_flag}/report"
    output_file = f"./{your_flag}/report/accuracy_results.xlsx"
    results = []
    result = subprocess.run(
        ['ls', '-1', folder_path],
        capture_output=True,
        text=True,
        check=True
    )

    all_files = result.stdout.strip().split('\n')

    json_files = [f for f in all_files if f.endswith('.json')]

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if "Core_Metrics" in data and "Accuracy" in data["Core_Metrics"]:
            accuracy = data["Core_Metrics"]["Accuracy"]
            results.append({
                "Filename": filename,
                "Accuracy": accuracy
            })

    if results:
        df = pd.DataFrame(results)
        df.to_excel(output_file, index=False)

if __name__ == "__main__":
    json_folder = "new-detector"
    extract_accuracy_to_excel(json_folder)