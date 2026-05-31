# MERIT-Assessor

This directory contains the stage-1 MERIT assessor training code based on verl.
The full verl framework is kept in this release; the main paper-facing entry points are:

- `verl/run_MERIT.sh`: GRPO training entry.
- `verl/data/data_process.py`: convert raw JSON data to verl parquet files.
- `verl/reward/reward.py`: custom reward and validation scoring logic.
- `verl/requirements.txt`: training dependencies.


## Usage

We recommend using a clean Conda environment. Our experiments were conducted with Python 3.12 and 4*NVIDIA A800-80GB GPUs.

## Environment

```bash
cd MERIT/MERIT-Assessor

conda create -n verl python=3.12 -y
conda activate verl

pip install -r verl/requirements.txt
```

Install the PyTorch/CUDA stack that matches your machine before installing vLLM if needed.

## Assets

Download the stage-1 backbone model:

- Backbone: https://huggingface.co/Qwen/Qwen3-4B
- Target path: `Models/Qwen3-4B/`

Download the released MERIT assessor checkpoint for evaluation or reuse:

- Checkpoint: https://huggingface.co/Luli3220/MERIT-4B-reviewer-assessor
- Target path: `Models/MERIT-4B-reviewer-assessor/`

Download the stage-1 training data:

- Dataset: https://huggingface.co/datasets/Luli3220/MERIT/tree/main/data/stage1_assessor/raw
- Target path: `verl/data/`
- Expected files: `train.json`, `validation.json`, `rubric.json`

## Judge API

The training reward uses an OpenAI-compatible judge API. Configure it before training:

```bash
export MERIT_JUDGE_API_KEY=your_api_key
export MERIT_JUDGE_BASE_URL=https://api.deepseek.com
export MERIT_JUDGE_MODEL=deepseek-reasoner
```

`DEEPSEEK_API_KEY` or `OPENAI_API_KEY` can also be used instead of `MERIT_JUDGE_API_KEY`.

## Data Processing

```bash
cd MERIT/MERIT-Assessor/verl
python data/data_process.py
```

This generates:

- `verl/data/train.parquet`
- `verl/data/validation.parquet`

Optional overrides:

```bash
python data/data_process.py \
  --train_file data/train.json \
  --validation_file data/validation.json \
  --rubric_file data/rubric.json \
  --output_dir data
```

## Training

```bash
cd MERIT/MERIT-Assessor/verl
bash run_MERIT.sh
```

Useful overrides:

```bash
ACTOR_MODEL=/path/to/Qwen3-4B \
DATA_DIR=/path/to/processed_data \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPU=4 \
bash run_MERIT.sh
```

Outputs are written under `verl/ckpts/`, `verl/logs/`, `verl/tensorboard_dir/`, and `verl/ray_tmp/` by default.

## Evaluation

The test split and evaluation script are under `test/`. The released checkpoint is expected at
`Models/MERIT-4B-reviewer-assessor/` by default.

```bash
cd MERIT/MERIT-Assessor

conda create -n eval python=3.12 -y
conda activate eval
pip install -r test/requirements.txt

cd test
python eval.py --device cuda:0
```

Useful overrides:

```bash
python eval.py \
  --model_path ../Models/MERIT-4B-reviewer-assessor \
  --data_path test.json \
  --output_dir outputs \
  --device cuda:0 \
  --batch_size 16
```

Evaluation outputs are written to `test/outputs/` by default.
