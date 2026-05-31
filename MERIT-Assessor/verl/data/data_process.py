import argparse
import json
import os
import sys
from pathlib import Path
import datasets

# Add repo root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import prompts locally to avoid heavy dependencies from verl package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import prompts
action_user_prompt = prompts.action_user_prompt
action_system_prompt = prompts.action_system_prompt

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_FILE = DATA_DIR / "train.json"
DEFAULT_VALIDATION_FILE = DATA_DIR / "validation.json"
DEFAULT_RUBRIC_FILE = DATA_DIR / "rubric.json"
DEFAULT_OUTPUT_DIR = DATA_DIR
DEFAULT_DATA_SOURCE = "MERIT-Assessor"
DEFAULT_ABILITY = "alignment"


def parse_args():
    parser = argparse.ArgumentParser(description="Build verl parquet data for MERIT-Assessor training.")
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation_file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--rubric_file", type=Path, default=DEFAULT_RUBRIC_FILE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_source", type=str, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--ability", type=str, default=DEFAULT_ABILITY)
    return parser.parse_args()


def load_rubric_map(rubric_path: str) -> dict:
    with open(rubric_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rubric_map = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        rubric = item.get("rubric")
        if isinstance(title, str) and title:
            rubric_map[title] = rubric
    return rubric_map


def build_candidate_history(author_papers) -> str:
    if not isinstance(author_papers, list) or not author_papers:
        return "None"
    blocks = []
    for idx, paper in enumerate(author_papers, 1):
        if not isinstance(paper, dict):
            continue
        title = paper.get("title") or ""
        abstract = paper.get("abstract") or ""
        content = f"{idx}. {title}".strip()
        if abstract:
            content = f"{content}\n{abstract}"
        if content:
            blocks.append(content)
    return "\n\n".join(blocks) if blocks else "None"


def fill_template(template: str, mapping: dict) -> str:
    result = template
    for key, value in mapping.items():
        result = result.replace(f"{{{{{key}}}}}", value or "")
    return result


def process_example(example, idx, split: str, rubric_map: dict, data_source: str, ability: str):
    paper = example.get("paper", {})
    author = example.get("author", {})
    paper_title = paper.get("title", "")
    paper_abstract = paper.get("abstract", "")
    paper_introduction = paper.get("introduction", "")
    author_papers = author.get("papers", [])

    candidate_history_list = build_candidate_history(author_papers)
    user_content = fill_template(
        action_user_prompt,
        {
            "paper_title": paper_title or "",
            "paper_abstract": paper_abstract or "",
            "paper_introduction": paper_introduction or "",
            "candidate_history_list": candidate_history_list,
        },
    ).strip()

    prompt = [
        {"role": "system", "content": action_system_prompt.strip()},
        {"role": "user", "content": user_content},
    ]

    rubric = rubric_map.get(paper_title, [])

    if split == "train":
        reward_model = {"style": "model", "ground_truth": rubric}
    else:
        # Pre-calculate label during data processing
        score_val = float(example["score"]) if "score" in example else 0
        label = 1 if score_val > 3 else 0
        reward_model = {"style": "model", "ground_truth": label}

    extra_info = {
        "split": split,
        "index": idx,
        "paper_title": paper_title or "",
        "paper_abstract": paper_abstract or "",
        "paper_introduction": paper_introduction or "",
        "candidate_history_list": candidate_history_list,
    }
    
    if split == "train":
        extra_info["rubric"] = rubric
    elif isinstance(example, dict) and "score" in example:
        # Map score to label: score > 3 -> 1, else 0
        score_val = float(example["score"])
        label = 1 if score_val > 3 else 0
        extra_info["label"] = label

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": reward_model,
        "extra_info": extra_info,
    }


if __name__ == "__main__":
    args = parse_args()
    rubric_map = load_rubric_map(args.rubric_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process Train
    print(f"Processing train file: {args.train_file}")
    train_dataset = datasets.load_dataset("json", data_files=str(args.train_file))["train"]
    train_dataset = train_dataset.map(
        function=lambda example, idx: process_example(
            example, idx, "train", rubric_map, args.data_source, args.ability
        ),
        with_indices=True,
        load_from_cache_file=False,
    )
    train_dataset.to_parquet(str(args.output_dir / "train.parquet"))

    # Process Validation
    print(f"Processing validation file: {args.validation_file}")
    val_dataset = datasets.load_dataset("json", data_files=str(args.validation_file))["train"]
    val_dataset = val_dataset.map(
        function=lambda example, idx: process_example(
            example, idx, "validation", rubric_map, args.data_source, args.ability
        ),
        with_indices=True,
        load_from_cache_file=False,
    )
    val_dataset.to_parquet(str(args.output_dir / "validation.parquet"))
    
    print("Processing complete.")
