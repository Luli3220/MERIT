#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=$(cd "$(dirname "$0")" && pwd)
RETRIEVER_DIR=$(cd "$BASE_DIR/../.." && pwd)

BACKBONE_MODEL_PATH="${BACKBONE_MODEL_PATH:-$RETRIEVER_DIR/Models/Qwen3-Embedding-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-$RETRIEVER_DIR/Models/MERIT-8B-retriever}"
DEVICE="${DEVICE:-cuda:0}"
PROFILE_TOP_K="${PROFILE_TOP_K:-3}"
START_ID="${START_ID:-1}"
END_ID="${END_ID:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TOPK_BATCH_SIZE="${TOPK_BATCH_SIZE:-32}"
PRED_PREFIX="${PRED_PREFIX:-ours_rc}"
TOPK_DIR="${TOPK_DIR:-$BASE_DIR/topk/${PRED_PREFIX}_topk${PROFILE_TOP_K}}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_DIR/predictions/${PRED_PREFIX}_topk${PROFILE_TOP_K}}"

for required_path in \
  "$BACKBONE_MODEL_PATH" \
  "$ADAPTER_PATH" \
  "$BASE_DIR/evaluation_datasets" \
  "$BASE_DIR/data/evaluations.csv"; do
  if [[ ! -e "$required_path" ]]; then
    echo "[Error] Missing required path: $required_path" >&2
    exit 1
  fi
done

python "$BASE_DIR/scripts/prepare_topk.py" \
  --model_path "$BACKBONE_MODEL_PATH" \
  --device "$DEVICE" \
  --base_dir "$BASE_DIR/evaluation_datasets" \
  --dataset_csv "$BASE_DIR/data/evaluations.csv" \
  --start "$START_ID" \
  --end "$END_ID" \
  --batch_size "$TOPK_BATCH_SIZE" \
  --profile_top_k "$PROFILE_TOP_K" \
  --topk_prefix "$PRED_PREFIX" \
  --save_topk_dir "$TOPK_DIR"

python "$BASE_DIR/scripts/ours.py" \
  --model_path "$BACKBONE_MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --device "$DEVICE" \
  --base_dir "$BASE_DIR/evaluation_datasets" \
  --dataset_csv "$BASE_DIR/data/evaluations.csv" \
  --start "$START_ID" \
  --end "$END_ID" \
  --batch_size "$BATCH_SIZE" \
  --topk_dir "$TOPK_DIR" \
  --prediction_prefix "$PRED_PREFIX" \
  --save_predictions_dir "$OUTPUT_DIR"

python "$BASE_DIR/evaluation_script.py" \
  --dataset "$BASE_DIR/data/evaluations.csv" \
  --prediction_dir "$OUTPUT_DIR" \
  --algo "$PRED_PREFIX"
