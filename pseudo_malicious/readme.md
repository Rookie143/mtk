# Pseudo-Malicious Detection Experiment

## Prerequisites

- Run this program in the same environment as `llm_mtk`.
- Place the complete Llama2-7b-chat-hf model weights, configuration, and tokenizer files in `llm/model/llama2/`.
- Ensure that `llm/datasets/reference_bank_data/` contains the following training data files:
  - `databricks-dolly-15k.txt`
  - `alpaca.txt`
  - `non_refusal_prompts_with_responses_80k.txt`
  - `AdvBench.txt`
  - `MaliciousInstruct.txt`
  - `PKU-SafeRLHF-prompts_3-6k.txt`
- The test data files are  in `llm/datasets/reference_bank_data/non_refusal_prompts_with_responses_80k.txt` and `llm/datasets/llam2/nanogcg_1.json`.
- To ensure that the pseudo-malicious prompts used to construct the reference bank do not overlap with those used for testing, we separately selected a set of non-overlapping prompts for reference bank construction. These prompts are stored in dataset/reference_bank_pmp_samples.txt in the current directory.

The model and training data paths above are relative to the repository root.

## Running the Experiment

From the repository root, enter this directory, install the dependencies provided in the `llm` directory, and run the entry script:

```bash
cd pseudo_malicious
python -m pip install -r ../llm/requirements.txt
python pseudo_malicious.py
```

The program performs data sampling, feature extraction, detector training, and evaluation. Results are saved to `results/results.csv` under this directory, including the attack detection rate (`tpr`), false positive rate on ordinary benign prompts (`fpr`), false positive rate on PMP (`pmp_fpr`), and F1 score (`f1`). Running the program again overwrites the result file.
