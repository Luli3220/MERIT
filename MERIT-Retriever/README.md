# MERIT-Retriever

This directory contains the official code for the MERIT retriever stage, including training and evaluation.

## Project Structure

```
├── Models/                  # Local backbone and released retriever checkpoints
├── train/
│   ├── data/
│   │   ├── raw/             # Raw stage-2 training data
│   ├── preprocess_topk.py   # Data preprocessing script
│   ├── train_contrastive.py # Contrastive LoRA training code
│   ├── run_contrastive.sh   # Training launch script
│   └── requirements.txt     # Requirements for training/evaluation
├── test/
│   ├── LR/                  # LR benchmark evaluation
│   └── CMU/                 # CMU benchmark evaluation
└── README.md                # This file
```

## Usage

We recommend using a clean Conda environment. Our experiments were conducted with Python 3.12 and two NVIDIA A800-80GB GPUs.

### Create Conda Environment

```
conda create -n merit-retriever python=3.12
conda activate merit-retriever
```

### Install Dependencies

```
cd train
pip install -r requirements.txt
```

### Prepare Data and Model

Download the Qwen3 embedding backbone from [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) and place it under:

```
Models/Qwen3-Embedding-8B/
```

Download the stage-2 retriever training data from [Luli3220/MERIT](https://huggingface.co/datasets/Luli3220/MERIT), folder `data/stage2_retriever/raw`, and place the files under:

```
train/data/raw/train_pc.json
train/data/raw/train_rc.json
train/data/raw/RATE_pc_val.json
train/data/raw/RATE_rc_val.json
```

Download the released stage-2 retriever checkpoint from [Luli3220/MERIT-8B-retriever](https://huggingface.co/Luli3220/MERIT-8B-retriever) and place it under:

```
Models/MERIT-8B-retriever/
```

You can override the backbone path with `MODEL_NAME=/path/to/Qwen3-Embedding-8B`.

## Training

Run the preprocessing script before training. The only exposed argument is `--top_k`, and the default value is 3.

```
cd train
python preprocess_topk.py --top_k 3
```

This script reads the raw files from `train/data/raw/` and writes:

```
train/data/processed/train_pc.json
train/data/processed/train_rc.json
train/data/processed/val_pc.json
train/data/processed/val_rc.json
```

Then run contrastive training:

```
bash run_contrastive.sh
```

The training script defaults to two GPUs:

```
CUDA_VISIBLE_DEVICES_VALUE=0,1
NPROC_PER_NODE=2
```


## Evaluation

### LR Benchmark

Run:

```
cd test/LR
bash eval_LR.sh
```

`eval_LR.sh` reads `test/LR/data/evaluations_pc.json` and `test/LR/data/evaluations_rc.json`, prepares top-k reviewer profiles, writes predictions to `test/LR/predictions/`, and prints pairwise Kendall loss and accuracy.

### CMU Benchmark

Run:

```
cd test/CMU
bash eval_CMU.sh
```

`eval_CMU.sh` evaluates `evaluation_datasets/d_20_1` through `evaluation_datasets/d_20_10`, writes top-k files to `test/CMU/topk/`, writes predictions to `test/CMU/predictions/`, and reports the official score with `evaluation_script.py`.
