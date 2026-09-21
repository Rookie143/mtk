# MTK: Defending Jailbreak Attacks on Large Language Models via Manifold Trajectory Kinetics

<p align="center">
  <b>Implementation of our USENIX Security paper</b>
</p>

<p align="center">
  <a href="https://www.usenix.org/conference/usenixsecurity26/presentation/zhang-hangtao">Paper</a> 
</p>

<p align="center">
  <img src="pipeline.pdf" width="90%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/USENIX-Security-blue">
  <img src="https://img.shields.io/badge/LLM-Jailbreak%20Detection-red">
  <img src="https://img.shields.io/badge/VLM-Supported-green">
  <img src="https://img.shields.io/badge/PyTorch-supported-orange">
</p>


## Repository Structure

| Path | Purpose |
| --- | --- |
| [`llm`](llm/llm_mtk) | MTK detection and evaluation scripts for LLMs. |
| [`vlm`](vlm) | MTK detection and evaluation scripts for VLMs. |
| [`adaptive_attack`](adaptive_attack) | Standalone adaptive attack against MTK with `L1`/`L2`/`L3` evasion losses. |


## 📋 Environment Requirements

### Basic Environment

- Python 3.8+ (3.9/3.10 recommended for PyTorch compatibility)
  
- CUDA 11.7+ (recommended for GPU acceleration; CPU is supported but slower)
  
- GPU Memory: ≥16GB for LLaVA-1.6-Vicuna-7B, ≥12GB for Qwen-VL-Chat
  

### Dependency Installation

```bash
# Clone repository
git clone https://github.com/Rookie143/mtk.git
cd mtk/vlm

# Install vlm core dependencies
pip install -r requirements.txt

#Install llm core dependencies
cd mtk/llm/llm_mtk
pip install -r requirements.txt
```

To install the adaptive attack package:

```bash
pip install -e ./adaptive_attack
```

## 🚀 VLM_Quick Start

### 1. Dataset Preparation

Place datasets in the specified paths (modify paths in `load_datasets.py`):

| Dataset | Path Example | Purpose |
| --- | --- | --- |
| [VQA](https://visualqa.org/download.html) | ./datasets/vqa/test2015 | Benign samples (training) |
| [MM-Vet v2](https://github.com/yuweihao/MM-Vet) | ./datasets/mm-vet-v2 | Benign samples (testing) |
| [SD-AdvBench] | ./datasets/sd_advbench | Malicious samples (training) |
| [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) | ./datasets/MM-SafetyBench | Malicious samples (testing) |
| [FigStep](https://github.com/CryptoAILab/FigStep/tree/main/data/images/SafeBench) | ./datasets/FigStep | Malicious samples (testing) |
| [JailBreakV_28K](https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k) | ./datasets/JailBreakV_28K | Malicious samples (testing) |
| [USB-Overrefusal](https://huggingface.co/datasets/cgjacklin/USB/tree/main) | ./datasets/MM-SafetyBench | Benign samples (testing) |

### 2. Model Weights Preparation

Download multimodal model weights and place them in the specified paths (modify `from_pretrained` paths in test scripts):

- LLaVA-1.6-Vicuna-7B: `./models/llava-v1.6-vicuna-7b-hf`
  
- Qwen-VL-Chat: `./model/qwen_vl_chat`
  

### 3. Run Detection

```bash
# Run jailbreak detection evaluation for LLaVA
python test_AUROC_llava.py 

# Run jailbreak detection evaluation for Qwen-VL
python test_AUROC_qwen.py
```

## 🚀 LLM_Quick Start

### 1. Training Datasets Preparation

Place **training datasets** in ./datasets/train_data:

| Dataset | Path Example | Purpose |
| --- | --- | --- |
| [Alpaca ](https://huggingface.co/datasets/gbharti/finance-alpaca) | ./datasets/train_data | Benign samples |
| [Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ./datasets/train_data | Benign samples |
| [Databricks-Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | ./datasets/train_data | Benign samples |
| [Or-Bench_80k](https://huggingface.co/datasets/bench-llm/or-bench) | ./datasets/train_data | Pseudo-Malicious samples |
| [MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct) | ./datasets/train_data | Malicious samples |
| [Advbench](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench) | ./datasets/train_data | Malicious samples |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF) | ./datasets/train_data | Malicious samples |

> ⚠️ Note: All datasets should be converted into a unified .txt format.

---

### 2. Test Datasets Preparation

Place **test datasets** under:

`./datasets/{model_name}_test`

where `{model_name}` must be one of:

- `llama2`
- `llama3`
- `mistral`
- `vicuna`

#### File Naming Convention

- **Jailbreak / malicious attack samples**:
  
  `{attack_name}_1.json`
  
- **Benign samples**:
  
  `{benign_name}_0.json`
  

Here, the suffix `_1` indicates malicious/jailbreak data, and `_0` indicates benign data.  
This naming convention is required for correct label parsing during evaluation.

### 3. Model Weights Preparation

Download the following large language model weights and place them under:

`./model`

Required models:

- **Llama2-7b-chat-hf**
  
- **Llama3-8b-Instruct**
  
- **Mistral-7b-instruct-v0.2**
  
- **Vicuna-7b-v1.5**
  

Each model should be stored in its own subdirectory following the Hugging Face standard structure.

### 4. Run Detection

```bash
# Run jailbreak detection evaluation for llama2
bash detection_llama2.bash

# Run jailbreak detection evaluation for llama3
bash detection_llama3.bash

# Run jailbreak detection evaluation for mitstral
bash detection_mistral.bash

# Run jailbreak detection evaluation for vicuna
bash detection_vicuna.bash

```
