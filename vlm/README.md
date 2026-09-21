# MTK VLM Detector

This directory contains the VLM evaluation scripts for MTK-style jailbreak detection.

The code trains a lightweight Isolation Forest detector on rank-sequence features extracted from benign and malicious VLM prompts, then reports AUPRC/AUROC and threshold-based FPR/precision on evaluation datasets.

## Installation

Run commands from this directory:

```bash
cd vlm
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the generic `torch` package is not suitable for your machine.

## Data layout

`sd_advbench` is included with this repository. Put the other datasets under `vlm/datasets/` with this layout:

```text
datasets/
  vqa/
    OpenEnded_mscoco_test2015_questions.json
    test2015/
  usb/
    overfuse_data.csv
  sd_advbench/
    prompt_img_map.csv
    outputs_new/
  mm-vet-v2/
    mm-vet-v2.json
    non_palette_images/
  MM-SafetyBench/
    data/
    image/
  FigStep/
    data/images/SafeBench/
  JailBreakV_28K/
    JailBreakV_28K.csv
```

Generated image caches, feature banks, and result files are ignored by git.

## Run LLaVA

Use either a Hugging Face model id or a local model path:

```bash
CUDA_VISIBLE_DEVICES=0 python test_AUROC_llava.py llava_run \
  --model-name-or-path llava-hf/llava-v1.6-vicuna-7b-hf \
  --output-dir experimental_results/llava_run
```

## Run Qwen-VL

```bash
CUDA_VISIBLE_DEVICES=0 python test_AUROC_qwen.py qwen_run \
  --model-name-or-path Qwen/Qwen-VL-Chat \
  --output-dir experimental_results/qwen_run
```

For offline use, replace `--model-name-or-path` with your local model directory.

## Outputs

Each run writes:

```text
experimental_results/<run_name>/
  saved_features_and_labels.pt
  training_sequences.pt
  results/
    test_<model>_<dataset>.csv
    <model>_AUROC_result.csv
```

Detector scores are reported as anomaly scores: larger values indicate more anomalous inputs.

## Useful options

```text
--k-nb                 number of nearest benign anchors used in the rank feature
--n-estimators         Isolation Forest tree count
--max-samples          Isolation Forest max_samples
--target-fpr           target FPR for threshold calibration on benign samples
--device-map           transformers device_map, e.g. auto or cuda
--output-dir           where to save feature banks and result files
```
