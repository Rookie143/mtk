#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets}"
TMP_DIR="${TMP_DIR:-${DATA_DIR}/.downloads}"

FAILED=0

log() {
  echo "[download_datasets] $*"
}

warn() {
  echo "[download_datasets][warn] $*" >&2
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    warn "missing command: $1"
    return 1
  fi
}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    log "skip existing file: $output"
    return 0
  fi

  log "download: $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "${output}.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}.part" "$url"
  else
    warn "curl or wget is required to download $url"
    FAILED=1
    return 1
  fi
  mv "${output}.part" "$output"
}

unzip_into() {
  local archive="$1"
  local output_dir="$2"
  need_cmd unzip || {
    FAILED=1
    return 1
  }
  mkdir -p "$output_dir"
  log "unzip: $archive -> $output_dir"
  unzip -n -q "$archive" -d "$output_dir"
}

hf_download() {
  local repo_type="$1"
  local repo_id="$2"
  local output_dir="$3"
  shift 3

  local hf_cli="${HF_CLI:-}"
  if [[ -z "$hf_cli" ]]; then
    if command -v hf >/dev/null 2>&1; then
      hf_cli="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      hf_cli="huggingface-cli"
    fi
  fi

  if [[ -z "$hf_cli" ]]; then
    warn "Hugging Face CLI is required for ${repo_id}. Install with: pip install huggingface_hub"
    FAILED=1
    return 1
  fi

  mkdir -p "$output_dir"
  log "huggingface download: ${repo_id} -> ${output_dir}"
  if ! "$hf_cli" download "$repo_id" \
    --repo-type "$repo_type" \
    --local-dir "$output_dir" \
    --local-dir-use-symlinks False \
    "$@"; then
    warn "failed to download ${repo_id}. If this is a gated dataset, run 'huggingface-cli login' and accept the dataset terms on Hugging Face first."
    FAILED=1
    return 1
  fi
}

download_vqa() {
  log "prepare VQA test2015"
  local vqa_dir="${DATA_DIR}/vqa"
  local question_zip="${TMP_DIR}/vqa/Questions_Test_mscoco.zip"
  local image_zip="${TMP_DIR}/vqa/test2015.zip"

  download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip" "$question_zip" || return 1
  unzip_into "$question_zip" "$vqa_dir" || return 1

  download_file "http://images.cocodataset.org/zips/test2015.zip" "$image_zip" || return 1
  unzip_into "$image_zip" "$vqa_dir" || return 1

  if [[ ! -f "${vqa_dir}/OpenEnded_mscoco_test2015_questions.json" ]]; then
    warn "VQA question file was not found at ${vqa_dir}/OpenEnded_mscoco_test2015_questions.json"
    FAILED=1
  fi
}

download_mm_vet() {
  log "prepare MM-Vet v2"
  local archive="${TMP_DIR}/mm-vet-v2/mm-vet-v2.zip"
  local extract_dir="${TMP_DIR}/mm-vet-v2/extracted"
  local out_dir="${DATA_DIR}/mm-vet-v2"

  download_file "https://github.com/yuweihao/MM-Vet/releases/download/v2/mm-vet-v2.zip" "$archive" || return 1
  unzip_into "$archive" "$extract_dir" || return 1
  mkdir -p "$out_dir"
  if [[ -d "${extract_dir}/mm-vet-v2" ]]; then
    cp -a "${extract_dir}/mm-vet-v2/." "$out_dir/"
  else
    cp -a "${extract_dir}/." "$out_dir/"
  fi
  if [[ -d "${out_dir}/images" && ! -e "${out_dir}/non_palette_images" ]]; then
    ln -s images "${out_dir}/non_palette_images"
  fi
  if [[ ! -f "${out_dir}/mm-vet-v2.json" || ! -e "${out_dir}/non_palette_images" ]]; then
    warn "MM-Vet v2 expected files were not found under ${out_dir}"
    FAILED=1
  fi
}

download_mm_safetybench() {
  log "prepare MM-SafetyBench"
  hf_download "dataset" "PKU-Alignment/MM-SafetyBench" "${DATA_DIR}/MM-SafetyBench" --include "data/**" --include "README.md" || return 1
  mkdir -p "${DATA_DIR}/MM-SafetyBench/image"
}

download_figstep() {
  log "prepare FigStep SafeBench images"
  local repo_dir="${TMP_DIR}/FigStep"
  local out_dir="${DATA_DIR}/FigStep"

  need_cmd git || {
    FAILED=1
    return 1
  }

  if [[ -d "${repo_dir}/.git" ]]; then
    git -C "$repo_dir" pull --ff-only
  else
    mkdir -p "$(dirname "$repo_dir")"
    git clone --depth 1 https://github.com/CryptoAILab/FigStep.git "$repo_dir"
  fi

  mkdir -p "${out_dir}/data"
  if [[ -d "${repo_dir}/data/images" ]]; then
    cp -a "${repo_dir}/data/images" "${out_dir}/data/"
  else
    warn "FigStep images folder was not found in ${repo_dir}/data/images"
    FAILED=1
  fi
  if [[ -d "${repo_dir}/data/question" ]]; then
    cp -a "${repo_dir}/data/question" "${out_dir}/data/"
  fi
}

download_jailbreakv() {
  log "prepare JailBreakV-28K"
  local out_dir="${DATA_DIR}/JailBreakV_28K"

  hf_download "dataset" "JailbreakV-28K/JailBreakV-28k" "$out_dir" --include "JailBreakV_28K/**" || return 1

  if [[ -d "${out_dir}/JailBreakV_28K" ]]; then
    cp -a "${out_dir}/JailBreakV_28K/." "$out_dir/"
  fi

  warn "JailBreakV-28K may require additional manually approved image downloads. If evaluation later reports missing image paths, follow the dataset page instructions and place the image folders under ${out_dir}."
}

download_usb() {
  log "prepare USB-Overrefusal"
  warn "cgjacklin/USB is gated on Hugging Face. Accept the dataset terms and run 'huggingface-cli login' before using this target."
  hf_download "dataset" "cgjacklin/USB" "${DATA_DIR}/usb" --include "USB_base_final.csv" --include "USB_hard_final.csv" --include "img.zip" --include "README.md" || return 1
  if [[ -f "${DATA_DIR}/usb/USB_base_final.csv" && ! -f "${DATA_DIR}/usb/overfuse_data.csv" ]]; then
    cp "${DATA_DIR}/usb/USB_base_final.csv" "${DATA_DIR}/usb/overfuse_data.csv"
  fi
  if [[ -f "${DATA_DIR}/usb/img.zip" ]]; then
    unzip_into "${DATA_DIR}/usb/img.zip" "${DATA_DIR}/usb" || return 1
  fi
  if [[ ! -f "${DATA_DIR}/usb/overfuse_data.csv" ]]; then
    warn "USB overfuse_data.csv was not prepared under ${DATA_DIR}/usb"
    FAILED=1
  fi
}

check_sd_advbench() {
  log "check bundled SD-AdvBench"
  if [[ ! -f "${DATA_DIR}/sd_advbench/prompt_img_map.csv" ]]; then
    warn "missing ${DATA_DIR}/sd_advbench/prompt_img_map.csv"
    FAILED=1
  fi
  if [[ ! -d "${DATA_DIR}/sd_advbench/outputs_new" ]]; then
    warn "missing ${DATA_DIR}/sd_advbench/outputs_new"
    FAILED=1
  fi
}

usage() {
  cat <<'EOF'
Usage:
  bash download_datasets.sh [target ...]

Targets:
  all              Download/check every dataset below, including gated USB.
  all-public       Download/check public datasets only, excluding gated USB.
  vqa              VQA test2015 questions and COCO test2015 images.
  mm-vet           MM-Vet v2 metadata and images.
  mm-safetybench   MM-SafetyBench parquet files.
  figstep          FigStep SafeBench images.
  jailbreakv       JailBreakV-28K files available from Hugging Face.
  usb              USB-Overrefusal dataset. Requires Hugging Face access approval.
  sd-advbench      Check bundled SD-AdvBench files.

Environment variables:
  DATA_DIR=/path/to/vlm/datasets
  TMP_DIR=/path/to/download/cache
  HF_CLI=/path/to/huggingface-cli

Examples:
  bash download_datasets.sh all
  bash download_datasets.sh vqa mm-vet mm-safetybench
EOF
}

mkdir -p "$DATA_DIR" "$TMP_DIR"

if [[ "$#" -eq 0 ]]; then
  set -- all
fi

for target in "$@"; do
  case "$target" in
  all)
      download_vqa
      download_mm_vet
      download_mm_safetybench
      download_figstep
      download_jailbreakv
      download_usb
      check_sd_advbench
      ;;
    all-public)
      download_vqa
      download_mm_vet
      download_mm_safetybench
      download_figstep
      download_jailbreakv
      check_sd_advbench
      ;;
    vqa)
      download_vqa
      ;;
    mm-vet)
      download_mm_vet
      ;;
    mm-safetybench)
      download_mm_safetybench
      ;;
    figstep)
      download_figstep
      ;;
    jailbreakv)
      download_jailbreakv
      ;;
    usb)
      download_usb
      ;;
    sd-advbench)
      check_sd_advbench
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      warn "unknown target: $target"
      usage
      exit 2
      ;;
  esac
done

if [[ "$FAILED" -ne 0 ]]; then
  warn "finished with missing or failed datasets; see warnings above."
  exit 1
fi

log "done. Data directory: ${DATA_DIR}"
