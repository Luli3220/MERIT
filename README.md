# MERIT: Matching Expertise via Rubric-Informed Training for Reviewer Assignment

This repository contains the official code release for MERIT. The system is organized into two stages:

- [`MERIT-Assessor`](MERIT-Assessor/): stage-1 reviewer assessor training and evaluation.
- [`MERIT-Retriever`](MERIT-Retriever/): stage-2 retriever training and evaluation.

Each module has its own README with detailed commands. This top-level README gives the full project layout and the shortest path to reproduce training and evaluation.

## Project Structure

```text
MERIT/
+-- MERIT-Assessor/
|   +-- Models/                 # Qwen3-4B backbone and released assessor checkpoint
|   +-- verl/                   # Stage-1 GRPO training code based on verl
|   +-- test/                   # Stage-1 assessor evaluation
|   +-- README.md
+-- MERIT-Retriever/
|   +-- Models/                 # Qwen3-Embedding-8B backbone and retriever checkpoint
|   +-- train/                  # Stage-2 retriever training
|   +-- test/                   # LR and CMU evaluation
|   +-- README.md
+-- README.md
```

## Assets

Download models and data from Hugging Face and place them under the expected local directories.

### Stage 1: Assessor

- Backbone: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
- Released checkpoint: [Luli3220/MERIT-4B-reviewer-assessor](https://huggingface.co/Luli3220/MERIT-4B-reviewer-assessor)
- Training data: [Luli3220/MERIT stage1_assessor/raw](https://huggingface.co/datasets/Luli3220/MERIT/tree/main/data/stage1_assessor/raw)

Expected local paths:

```text
MERIT-Assessor/Models/Qwen3-4B/
MERIT-Assessor/Models/MERIT-4B-reviewer-assessor/
MERIT-Assessor/verl/data/train.json
MERIT-Assessor/verl/data/validation.json
MERIT-Assessor/verl/data/rubric.json
```

### Stage 2: Retriever

- Backbone: [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
- Released checkpoint: [Luli3220/MERIT-8B-retriever](https://huggingface.co/Luli3220/MERIT-8B-retriever)
- Training data: [Luli3220/MERIT stage2_retriever/raw](https://huggingface.co/datasets/Luli3220/MERIT/tree/main/data/stage2_retriever/raw)

Expected local paths:

```text
MERIT-Retriever/Models/Qwen3-Embedding-8B/
MERIT-Retriever/Models/MERIT-8B-retriever/
MERIT-Retriever/train/data/raw/train_pc.json
MERIT-Retriever/train/data/raw/train_rc.json
MERIT-Retriever/train/data/raw/RATE_pc_val.json
MERIT-Retriever/train/data/raw/RATE_rc_val.json
```

## Stage 1: MERIT-Assessor

The assessor is trained with GRPO using verl. The reward function uses an OpenAI-compatible judge API during training.

### Environment

```bash
cd MERIT/MERIT-Assessor

conda create -n verl python=3.12 -y
conda activate verl
pip install -r verl/requirements.txt
```

### Judge API

```bash
export MERIT_JUDGE_API_KEY=your_api_key
export MERIT_JUDGE_BASE_URL=https://api.deepseek.com
export MERIT_JUDGE_MODEL=deepseek-reasoner
```

`DEEPSEEK_API_KEY` or `OPENAI_API_KEY` can also be used instead of `MERIT_JUDGE_API_KEY`.

### Data Processing

```bash
cd MERIT/MERIT-Assessor/verl
python data/data_process.py
```

This generates:

```text
MERIT-Assessor/verl/data/train.parquet
MERIT-Assessor/verl/data/validation.parquet
```

### Training

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

### Evaluation

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

For more details, see [MERIT-Assessor/README.md](MERIT-Assessor/README.md).

## Stage 2: MERIT-Retriever

The retriever is trained with contrastive learning on top of `Qwen3-Embedding-8B`.

### Environment

```bash
cd MERIT/MERIT-Retriever

conda create -n merit-retriever python=3.12 -y
conda activate merit-retriever

cd train
pip install -r requirements.txt
```

### Preprocessing

```bash
cd MERIT/MERIT-Retriever/train
python preprocess_topk.py --top_k 3
```

This generates:

```text
MERIT-Retriever/train/data/processed/train_pc.json
MERIT-Retriever/train/data/processed/train_rc.json
MERIT-Retriever/train/data/processed/val_pc.json
MERIT-Retriever/train/data/processed/val_rc.json
```

### Training

```bash
cd MERIT/MERIT-Retriever/train
bash run_contrastive.sh
```

Useful overrides:

```bash
MODEL_NAME=/path/to/Qwen3-Embedding-8B \
CUDA_VISIBLE_DEVICES_VALUE=0,1 \
NPROC_PER_NODE=2 \
bash run_contrastive.sh
```

### Evaluation

LR benchmark:

```bash
cd MERIT/MERIT-Retriever/test/LR
bash eval_LR.sh
```

CMU benchmark:

```bash
cd MERIT/MERIT-Retriever/test/CMU
bash eval_CMU.sh
```

Useful overrides:

```bash
BACKBONE_MODEL_PATH=/path/to/Qwen3-Embedding-8B \
ADAPTER_PATH=/path/to/MERIT-8B-retriever \
DEVICE=cuda:0 \
bash eval_LR.sh
```

For more details, see [MERIT-Retriever/README.md](MERIT-Retriever/README.md).

## Outputs

Generated files are ignored by git by default:

- `MERIT-Assessor/verl/ckpts/`
- `MERIT-Assessor/verl/logs/`
- `MERIT-Assessor/verl/tensorboard_dir/`
- `MERIT-Assessor/test/outputs/`
- `MERIT-Retriever/train/outputs/`
- `MERIT-Retriever/train/data/processed/`
- `MERIT-Retriever/test/LR/predictions/`
- `MERIT-Retriever/test/CMU/topk/`
- `MERIT-Retriever/test/CMU/predictions/`

