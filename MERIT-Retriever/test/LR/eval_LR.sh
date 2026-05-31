#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
RETRIEVER_DIR=$(cd "$BASE_DIR/../.." && pwd)

BACKBONE_MODEL_PATH="${BACKBONE_MODEL_PATH:-$RETRIEVER_DIR/Models/Qwen3-Embedding-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$RETRIEVER_DIR/Models/MERIT-8B-retriever}"
DATA_DIR="${DATA_DIR:-$BASE_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/predictions}"
TOPK_DIR="${TOPK_DIR:-$OUTPUT_DIR/topk}"
DEVICE="${DEVICE:-cuda:1}"
PROFILE_TOP_K="${PROFILE_TOP_K:-3}"
TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-64}"

for required_path in \
  "$BACKBONE_MODEL_PATH" \
  "$ADAPTER_PATH" \
  "$DATA_DIR/evaluations_pc.json" \
  "$DATA_DIR/evaluations_rc.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[Error] Missing required path: $required_path" >&2
    exit 1
  fi
done

python "$BASE_DIR/prepare_topk_LR.py" \
  --model_path "$BACKBONE_MODEL_PATH" \
  --data_path "$DATA_DIR/evaluations_pc.json" "$DATA_DIR/evaluations_rc.json" \
  --output_path "$TOPK_DIR" \
  --profile_top_k "$PROFILE_TOP_K" \
  --batch_size "$TOPK_BATCH_SIZE" \
  --device "$DEVICE"

python "$BASE_DIR/eval_LR.py" \
  --model_path "$BACKBONE_MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --data_path "$TOPK_DIR/RPA_pc_topk.json" "$TOPK_DIR/RPA_rc_topk.json" \
  --output_path "$OUTPUT_DIR" \
  --device "$DEVICE"
