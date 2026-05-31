#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSESSOR_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${ASSESSOR_DIR}:${PYTHONPATH:-}"

# --- Configuration ---
# Hardware
num_gpu="${NUM_GPU:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"


# Model & Data Paths
ACTOR_MODEL="${ACTOR_MODEL:-${ASSESSOR_DIR}/Models/Qwen3-4B}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
CUSTOM_REWARD_PATH="${CUSTOM_REWARD_PATH:-${SCRIPT_DIR}/reward/reward.py}"

# Project & Experiment
trainer_project_name="${TRAINER_PROJECT_NAME:-MERIT-Assessor}"
trainer_experiment_name="${TRAINER_EXPERIMENT_NAME:-grpo}"
CKPTS_DIR="${CKPTS_DIR:-${SCRIPT_DIR}/ckpts/${trainer_project_name}/${trainer_experiment_name}}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${SCRIPT_DIR}/tensorboard_dir/${trainer_project_name}/${trainer_experiment_name}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${trainer_project_name}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${trainer_experiment_name}.log}"
RAY_TMP_DIR="${RAY_TMP_DIR:-${SCRIPT_DIR}/ray_tmp}"
mkdir -p "${CKPTS_DIR}" "${TENSORBOARD_DIR}" "${LOG_DIR}" "${RAY_TMP_DIR}"

# Training Hyperparameters
train_batch_size="${TRAIN_BATCH_SIZE:-16}"
max_prompt_length="${MAX_PROMPT_LENGTH:-$((1024 * 5))}"
max_response_length="${MAX_RESPONSE_LENGTH:-$((1024 * 4))}"
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
lr="${LR:-5e-7}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
rollout_n="${ROLLOUT_N:-8}"
total_epochs="${TOTAL_EPOCHS:-2}"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}"
temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"
val_temperature="${VAL_TEMPERATURE:-0.0}"
val_top_p="${VAL_TOP_P:-0.7}"



# --- End Configuration ---

if [[ ! -d "${ACTOR_MODEL}" ]]; then
    echo "Missing actor model: ${ACTOR_MODEL}" >&2
    echo "Download Qwen/Qwen3-4B to MERIT-Assessor/Models/Qwen3-4B or set ACTOR_MODEL." >&2
    exit 1
fi

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/validation.parquet" ]]; then
    echo "Missing processed data under ${DATA_DIR}." >&2
    echo "Run: python data/data_process.py" >&2
    exit 1
fi

if [[ ! -f "${CUSTOM_REWARD_PATH}" ]]; then
    echo "Missing reward file: ${CUSTOM_REWARD_PATH}" >&2
    exit 1
fi

# 3. Start GRPO Training
echo "Starting GRPO Training on ${num_gpu} GPUs..."

export NCCL_P2P_DISABLE=1      
export NCCL_SHM_DISABLE=0      
export NCCL_IB_DISABLE=1       
export NCCL_TIMEOUT=1200000        
export RAY_PPO_LOG_LEVEL=info     

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/validation.parquet \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${ACTOR_MODEL} \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${num_gpu} \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger=['console','tensorboard'] \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=${num_gpu} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=${total_epochs} \
    trainer.val_before_train=False \
    trainer.max_actor_ckpt_to_keep=10 \
    custom_reward_function.path=${CUSTOM_REWARD_PATH} \
    custom_reward_function.name=compute_score \
    trainer.val_before_train=True \
    +ray_kwargs.ray_init._temp_dir=${RAY_TMP_DIR} 2>&1 | tee "${LOG_PATH}"
