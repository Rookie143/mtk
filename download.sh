#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DOWNLOAD_DIR=${DOWNLOAD_DIR:-"$ROOT_DIR/.downloads"}
DATA_DIR=${DATA_DIR:-"$ROOT_DIR/vlm/datasets"}

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$1" >&2
        exit 1
    fi
}

has_content() {
    [ -d "$1" ] && [ -n "$(find "$1" -mindepth 1 -maxdepth 1 2>/dev/null)" ]
}

if command -v python3 >/dev/null 2>&1; then
    PYTHON=${PYTHON:-python3}
elif command -v python >/dev/null 2>&1; then
    PYTHON=${PYTHON:-python}
else
    printf 'Missing required command: python3\n' >&2
    exit 1
fi

need_cmd git
need_cmd wget
need_cmd unzip

"$PYTHON" - <<'PY'
import importlib.util
import sys

missing = []
for module in ("huggingface_hub", "pandas", "pyarrow", "PIL"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)

if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    print("Install them with: python3 -m pip install -U huggingface_hub pandas pyarrow pillow", file=sys.stderr)
    sys.exit(1)
PY

mkdir -p "$DOWNLOAD_DIR" "$DATA_DIR"
cd "$DATA_DIR"

clone_or_update() {
    repo_url=$1
    target_dir=$2

    if [ -d "$target_dir/.git" ]; then
        log "Updating $target_dir"
        git -C "$target_dir" pull --ff-only
    elif has_content "$target_dir"; then
        log "Skipping $target_dir because it already exists and is not empty"
    else
        log "Cloning $repo_url into $target_dir"
        git clone "$repo_url" "$target_dir"
    fi
}

hf_snapshot() {
    repo_id=$1
    repo_type=$2
    local_dir=$3
    allow_patterns=${4:-}

    mkdir -p "$local_dir"
    HF_REPO_ID="$repo_id" \
    HF_REPO_TYPE="$repo_type" \
    HF_LOCAL_DIR="$local_dir" \
    HF_ALLOW_PATTERNS="$allow_patterns" \
    "$PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

allow_patterns = [item for item in os.environ.get("HF_ALLOW_PATTERNS", "").split(":") if item]
kwargs = {}
if allow_patterns:
    kwargs["allow_patterns"] = allow_patterns

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    repo_type=os.environ["HF_REPO_TYPE"],
    local_dir=os.environ["HF_LOCAL_DIR"],
    **kwargs,
)
PY
}

download_zip() {
    url=$1
    output=$2

    if [ -f "$output" ]; then
        log "Using cached $(basename "$output")"
    else
        log "Downloading $url"
        wget -c -O "$output" "$url"
    fi
}

download_figstep() {
    clone_or_update "https://github.com/CryptoAILab/FigStep.git" "FigStep"
}

download_jailbreakv() {
    if [ -f "JailBreakV_28K/JailBreakV_28K.csv" ] && has_content "JailBreakV_28K/query_related"; then
        log "Skipping JailBreakV_28K because it already exists"
    else
        log "Downloading JailBreakV_28K from Hugging Face"
        hf_snapshot "JailbreakV-28K/JailBreakV-28k" "dataset" "JailBreakV_28K"
    fi
}

download_mm_safetybench() {
    if has_content "MM-SafetyBench/data"; then
        log "Skipping MM-SafetyBench public parquet data because it already exists"
    else
        log "Downloading MM-SafetyBench public parquet data from Hugging Face"
        hf_snapshot "PKU-Alignment/MM-SafetyBench" "dataset" "MM-SafetyBench"
    fi
}

download_mm_vet_v2() {
    if [ -f "mm-vet-v2/mm-vet-v2.json" ] && has_content "mm-vet-v2/images" && has_content "mm-vet-v2/non_palette_images"; then
        log "Skipping mm-vet-v2 because it already exists"
        return
    fi

    log "Downloading MM-Vet v2 JSON from Hugging Face Space"
    hf_snapshot "whyu/MM-Vet-v2_Evaluator" "space" "$DOWNLOAD_DIR/MM-Vet-v2_Evaluator" "mm-vet-v2/mm-vet-v2.json"

    mkdir -p "mm-vet-v2"
    cp "$DOWNLOAD_DIR/MM-Vet-v2_Evaluator/mm-vet-v2/mm-vet-v2.json" "mm-vet-v2/mm-vet-v2.json"

    log "Downloading MM-Vet v2 image parquet from Hugging Face dataset"
    hf_snapshot "whyu/mm-vet-v2" "dataset" "$DOWNLOAD_DIR/mm-vet-v2-dataset"

    log "Reconstructing mm-vet-v2/images and mm-vet-v2/non_palette_images"
    DATA_DIR="$DATA_DIR" \
    MMVET_HF_DIR="$DOWNLOAD_DIR/mm-vet-v2-dataset" \
    "$PYTHON" - <<'PY'
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

data_dir = Path(os.environ["DATA_DIR"])
hf_dir = Path(os.environ["MMVET_HF_DIR"])
target = data_dir / "mm-vet-v2"
json_path = target / "mm-vet-v2.json"
images_dir = target / "images"
non_palette_dir = target / "non_palette_images"

images_dir.mkdir(parents=True, exist_ok=True)
non_palette_dir.mkdir(parents=True, exist_ok=True)

with json_path.open("r", encoding="utf-8") as handle:
    annotations = json.load(handle)

parquet_files = sorted((hf_dir / "data").glob("*.parquet"))
if not parquet_files:
    raise FileNotFoundError(f"No parquet files found under {hf_dir / 'data'}")

frame = pd.read_parquet(parquet_files[0])
image_columns = [column for column in frame.columns if column.startswith("image")]
if not image_columns:
    raise ValueError("No image columns found in MM-Vet v2 parquet data")

saved = 0
for row_index, row in frame.iterrows():
    key = str(row.get("id") or row.get("question_id") or row_index)
    record = annotations.get(key)
    if not record:
        continue

    filenames = re.findall(r"<IMG>([^<\s]+)", record.get("question", ""))
    for image_index, filename in enumerate(filenames):
        if image_index >= len(image_columns):
            continue

        value = row[image_columns[image_index]]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue

        output_path = images_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(value, dict):
            if value.get("bytes") is not None:
                output_path.write_bytes(value["bytes"])
            elif value.get("path") and Path(value["path"]).exists():
                shutil.copy2(value["path"], output_path)
            else:
                continue
        elif isinstance(value, (bytes, bytearray)):
            output_path.write_bytes(bytes(value))
        else:
            continue
        saved += 1

copied = 0
for image_path in sorted(images_dir.iterdir()):
    if not image_path.is_file():
        continue
    try:
        with Image.open(image_path) as image:
            if image.mode == "P":
                continue
        shutil.copy2(image_path, non_palette_dir / image_path.name)
        copied += 1
    except Exception as exc:
        print(f"Skipping unreadable image {image_path}: {exc}")

print(f"Saved {saved} MM-Vet v2 images")
print(f"Copied {copied} non-palette images")
PY
}

download_usb() {
    if [ -f "usb/overfuse_data.csv" ] && has_content "usb/img"; then
        log "Skipping usb because it already exists"
    else
        log "Downloading USB from Hugging Face"
        hf_snapshot "cgjacklin/USB" "dataset" "usb"
    fi
}

download_vqa() {
    if [ -f "vqa/OpenEnded_mscoco_test2015_questions.json" ] && has_content "vqa/test2015"; then
        log "Skipping vqa because it already exists"
        return
    fi

    mkdir -p "vqa"

    questions_zip="$DOWNLOAD_DIR/Questions_Test_mscoco.zip"
    test_images_zip="$DOWNLOAD_DIR/test2015.zip"

    download_zip "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Test_mscoco.zip" "$questions_zip"
    log "Extracting VQA questions"
    unzip -n "$questions_zip" -d "vqa"

    download_zip "https://images.cocodataset.org/zips/test2015.zip" "$test_images_zip"
    log "Extracting COCO test2015 images"
    unzip -n "$test_images_zip" -d "vqa"
}

download_figstep
download_jailbreakv
download_mm_safetybench
download_mm_vet_v2
download_usb
download_vqa

log "Done. sd_advbench was intentionally not downloaded."
