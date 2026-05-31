import argparse
import json
import os
import re
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from prompts import action_system_prompt, action_user_prompt


TEST_DIR = Path(__file__).resolve().parent
ASSESSOR_DIR = TEST_DIR.parent
DEFAULT_MODEL_PATH = ASSESSOR_DIR / "Models" / "MERIT-4B-reviewer-assessor"
DEFAULT_DATA_PATH = TEST_DIR / "test.json"
DEFAULT_OUTPUT_DIR = TEST_DIR / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MERIT stage-1 assessor.")
    parser.add_argument("--model_path", type=Path, default=Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    parser.add_argument("--data_path", type=Path, default=Path(os.getenv("DATA_PATH", DEFAULT_DATA_PATH)))
    parser.add_argument("--output_dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--device", type=str, default=os.getenv("DEVICE", "cuda:0"))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--max_prompt_length", type=int, default=int(os.getenv("MAX_PROMPT_LENGTH", str(1024 * 5))))
    parser.add_argument("--max_response_length", type=int, default=int(os.getenv("MAX_RESPONSE_LENGTH", str(1024 * 4))))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    return parser.parse_args()


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


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def remove_thinking(text):
    # Remove <think>...</think> blocks or the specific placeholder tag
    # Handles both standard </think> and the user-reported placeholder
    pattern = r"<think>.*?(?:</think>|<\[PLHD21_never_used_51bce0c785ca2f68081bfa7d91973934\]>)"
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()


def parse_label(output_text):
    # Remove thinking process first
    clean_text = remove_thinking(output_text)
    
    # Try to find [FINAL_LABEL] block
    if "[FINAL_LABEL]" in clean_text:
        # Extract everything after [FINAL_LABEL]
        after_label = clean_text.split("[FINAL_LABEL]")[-1]
        # Look for the first occurrence of 0 or 1
        match = re.search(r"\b(0|1)\b", after_label)
        if match:
            return int(match.group(1))
    
    # Fallback: look for <0 or 1> pattern specifically mentioned in prompt
    match = re.search(r"<\s*(0|1)\s*>", clean_text)
    if match:
        return int(match.group(1))
        
    return 0  # Default to 0 (conservative)


def build_output_path(model_path: Path, output_dir: Path) -> Path:
    step_match = re.search(r"step_(\d+)", str(model_path))
    if step_match:
        suffix = f"step_{step_match.group(1)}"
    else:
        suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_path.name.strip()) or "model"
    return output_dir / f"eval_results_{suffix}.json"


def save_results(output_file: Path, metrics: dict, predictions: list[int], ground_truths: list[int]) -> None:
    payload = {
        **metrics,
        "predictions": predictions,
        "ground_truths": ground_truths,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_metrics(ground_truths: list[int], predictions: list[int]) -> dict:
    return {
        "accuracy": accuracy_score(ground_truths, predictions),
        "balanced_accuracy": balanced_accuracy_score(ground_truths, predictions),
        "f1": f1_score(ground_truths, predictions, zero_division=0),
        "recall": recall_score(ground_truths, predictions, zero_division=0),
        "precision": precision_score(ground_truths, predictions, zero_division=0),
    }


def resolve_device_map(device: str):
    if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return device
    return {"": device}


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = build_output_path(args.model_path, args.output_dir)

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=resolve_device_map(args.device),
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path)
    
    prompts = []
    ground_truths = []
    
    print("Preparing prompts...")
    for item in data:
        # Ground Truth Logic: score > 3 -> 1, else 0
        try:
            score = float(item.get("score", 0))
        except (ValueError, TypeError):
            score = 0
        gt = 1 if score > 3 else 0
        
        # Prepare Prompt
        paper = item.get("paper", {})
        author = item.get("author", {})
        
        paper_title = paper.get("title", "")
        paper_abstract = paper.get("abstract", "")
        paper_introduction = paper.get("introduction", "")
        candidate_history = build_candidate_history(author.get("papers", []))
        abstract_block = f"Abstract:\n{paper_abstract}"
        introduction_block = f"Introduction:\n{paper_introduction}"
        
        user_content = action_user_prompt.replace("{{paper_title}}", paper_title) \
                                         .replace("{{paper_abstract}}", abstract_block) \
                                         .replace("{{paper_introduction}}", introduction_block) \
                                         .replace("{{candidate_history_list}}", candidate_history)
        
        messages = [
            {"role": "system", "content": action_system_prompt},
            {"role": "user", "content": user_content}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Check token length
        tokenized_len = len(tokenizer.encode(text))
        if tokenized_len > args.max_prompt_length:
            print(
                f"Skipping sample {item.get('id', 'unknown')} "
                f"due to length {tokenized_len} > {args.max_prompt_length}"
            )
            continue
            
        prompts.append(text)
        ground_truths.append(gt)

    predictions = []

    if not prompts:
        raise ValueError("No examples available after prompt length filtering.")

    print(f"Starting inference on {len(prompts)} examples using {args.device}...")
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i:i + args.batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=args.max_prompt_length
        ).to(args.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_response_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[:, input_len:]
        generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # Parse
        for text in generated_texts:
            pred = parse_label(text)
            predictions.append(pred)

        # Calculate current metrics
        current_ground_truths = ground_truths[:len(predictions)]
        current_metrics = compute_metrics(current_ground_truths, predictions)
        
        print(
            f"\nBatch {(i // args.batch_size) + 1}: "
            f"Current Accuracy: {current_metrics['accuracy']:.4f}, "
            f"Current Balanced Accuracy: {current_metrics['balanced_accuracy']:.4f}, "
            f"Current F1: {current_metrics['f1']:.4f}, "
            f"Current Recall: {current_metrics['recall']:.4f}, "
            f"Current Precision: {current_metrics['precision']:.4f}"
        )

        # Save results incrementally
        save_results(output_file, current_metrics, predictions, ground_truths)

    # Final Metrics
    metrics = compute_metrics(ground_truths, predictions)
    
    print("-" * 30)
    print("Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    gt_zero = sum(1 for x in ground_truths if x == 0)
    gt_one = sum(1 for x in ground_truths if x == 1)
    pred_zero = sum(1 for x in predictions if x == 0)
    pred_one = sum(1 for x in predictions if x == 1)
    print(f"Groundtruth counts: 0={gt_zero}, 1={gt_one}")
    print(f"Prediction counts: 0={pred_zero}, 1={pred_one}")
    print("-" * 30)
    
    # Save results
    save_results(output_file, metrics, predictions, ground_truths)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
