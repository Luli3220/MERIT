#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="${TRAIN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RETRIEVER_DIR="${RETRIEVER_DIR:-$(cd "$TRAIN_DIR/.." && pwd)}"
SCRIPT_PATH="${SCRIPT_PATH:-$TRAIN_DIR/train_contrastive.py}"
MODEL_NAME="${MODEL_NAME:-$RETRIEVER_DIR/Models/Qwen3-Embedding-8B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAIN_DIR/outputs}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29530}"

TRAIN_PC="${TRAIN_PC:-$TRAIN_DIR/data/processed/train_pc.json}"
TRAIN_RC="${TRAIN_RC:-$TRAIN_DIR/data/processed/train_rc.json}"
VAL_PC="${VAL_PC:-$TRAIN_DIR/data/processed/val_pc.json}"
VAL_RC="${VAL_RC:-$TRAIN_DIR/data/processed/val_rc.json}"

BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-4e-5}"
TEMPERATURE="${TEMPERATURE:-0.04}"
INBATCH_W="${INBATCH_W:-0.5}"
PAIR_W="${PAIR_W:-1.0}"
PAIR_MARGIN="${PAIR_MARGIN:-0.0}"
SEED="${SEED:-3407}"
MAX_LEN_Q="${MAX_LEN_Q:-2048}"
MAX_LEN_D="${MAX_LEN_D:-2048}"
PATIENCE="${PATIENCE:-50}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-10}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.06}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"

RUN_NAME="${RUN_NAME:-${INBATCH_W}_pairw${PAIR_W}_pm${PAIR_MARGIN}_tem${TEMPERATURE}_lr${LR}_acc${GRAD_ACCUM}_deepseek}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[Error] Training script not found: $SCRIPT_PATH" >&2
  exit 1
fi

for data_path in "$TRAIN_PC" "$TRAIN_RC" "$VAL_PC" "$VAL_RC"; do
  if [[ -n "$data_path" && ! -f "$data_path" ]]; then
    echo "[Error] Missing processed data: $data_path" >&2
    echo "        Run: python $TRAIN_DIR/preprocess_topk.py --top_k 3" >&2
    exit 1
  fi
done

mkdir -p "$OUTPUT_ROOT"
OUT_DIR="$OUTPUT_ROOT/ckpts/$RUN_NAME"
LOG_FILE="$OUTPUT_ROOT/ckpts/${RUN_NAME}.log"
mkdir -p "$OUT_DIR"

CMD=(
  accelerate launch
  --num_processes "$NPROC_PER_NODE"
  --multi_gpu
  --mixed_precision bf16
  --main_process_port "$MASTER_PORT"
  "$SCRIPT_PATH"
  --model_name "$MODEL_NAME"
  --output_dir "$OUT_DIR"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --grad_accum "$GRAD_ACCUM"
  --epochs "$EPOCHS"
  --lr "$LR"
  --temperature "$TEMPERATURE"
  --inbatch_weight "$INBATCH_W"
  --pair_weight "$PAIR_W"
  --warmup_ratio 0.01
  --pair_margin "$PAIR_MARGIN"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --attn_impl "$ATTN_IMPL"
  --max_len_q "$MAX_LEN_Q"
  --max_len_d "$MAX_LEN_D"
  --patience "$PATIENCE"
  --eval_every_steps "$EVAL_EVERY_STEPS"
  --seed "$SEED"
  --bf16
  --run_tag "$RUN_NAME"
)

if [[ -n "$TRAIN_PC" ]]; then CMD+=(--gold_train_pc_path "$TRAIN_PC"); fi
if [[ -n "$TRAIN_RC" ]]; then CMD+=(--gold_train_rc_path "$TRAIN_RC"); fi
if [[ -n "$VAL_PC" ]]; then CMD+=(--gold_val_pc_path "$VAL_PC"); fi
if [[ -n "$VAL_RC" ]]; then CMD+=(--gold_val_rc_path "$VAL_RC"); fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE"
printf 'CMD: %q ' "${CMD[@]}"
echo

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
