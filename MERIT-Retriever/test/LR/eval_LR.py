import argparse
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from evaluation_script import compute_pairwise_weighted_kendall_loss, compute_pairwise_accuracy

DEFAULT_MODEL_PATH = None

DEFAULT_TASK_PROMPT = "For the Paper (Query): Represent this academic paper's title and abstract for retrieving suitable reviewer profiles."
DEFAULT_TASK_PROMPT_RC = "For the Author (Query): Represent this author's publication history and expertise for finding relevant academic papers."

def _get_id(obj: dict) -> str:
    if not isinstance(obj, dict):
        return "unknown_id"
    return obj.get("paper_id") or obj.get("author_id") or obj.get("id") or "unknown_id"

def _ensure_dir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)

def _output_base_for_input(data_path: str) -> str:
    name = os.path.basename(data_path).lower()
    if "pc" in name:
        suffix = "pc"
    elif "rc" in name:
        suffix = "rc"
    else:
        suffix = os.path.splitext(os.path.basename(data_path))[0]
    return f"RPA_{suffix}"


def apply_instruction(text, instruction):
    if not instruction:
        return text
    if not text:
        return instruction
    return f"{instruction}\n{text}"

def build_profile_text(papers):
    chunks = []
    for idx, p in enumerate(papers, start=1):
        title = (p.get('title') or p.get('paper_title') or "").strip()
        abstract = (p.get('abstract') or "").strip()
        parts = []
        if title:
            parts.append(f"Title: {title}")
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if parts:
            chunks.append(f"Paper {idx}:\n" + "\n".join(parts))
    return "\n\n".join(chunks)


def build_profile_doc_text(papers):
    merged = build_profile_text(papers)
    return f"Reviewer profile:\n{merged}" if merged else "Reviewer profile:"

def get_paper_text(paper_data):
    title = (paper_data.get('paper_title') or paper_data.get('title') or "").strip()
    abstract = (paper_data.get('abstract') or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts)


def get_qwen3_query_prompt(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


def _resolve_rc_anchor_profiles(ex: dict) -> tuple[dict, dict, dict]:
    anchor = ex.get("anchor", {}) if isinstance(ex.get("anchor", {}), dict) else {}
    anchor_for_pos = ex.get("anchor_for_positive", {}) if isinstance(ex.get("anchor_for_positive", {}), dict) else {}
    anchor_for_neg = ex.get("anchor_for_negative", {}) if isinstance(ex.get("anchor_for_negative", {}), dict) else {}
    anchor_pos = anchor_for_pos if isinstance(anchor_for_pos.get("papers", []), list) and len(anchor_for_pos.get("papers", [])) > 0 else anchor
    anchor_neg = anchor_for_neg if isinstance(anchor_for_neg.get("papers", []), list) and len(anchor_for_neg.get("papers", [])) > 0 else anchor
    return anchor, anchor_pos, anchor_neg


def _detect_dataset_type(data: list) -> str:
    dataset_type = None
    for x in data[:5]:
        if isinstance(x, dict) and "type" in x:
            dataset_type = x.get("type")
            break
    if not dataset_type:
        if data and "anchor" in data[0]:
            anchor = data[0]["anchor"]
            if "papers" in anchor:
                dataset_type = "reviewer_centric"
            else:
                dataset_type = "paper_centric"
    return dataset_type
# --- Model Loading Logic ---

def load_st_model_with_adapter(model_path, adapter_path, device):
    from peft import PeftModel
    print(f"Loading base model from {model_path}...")
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
    
    print(f"Loading adapter from {adapter_path}...")
    # 获取 SentenceTransformer 内部的 transformer 模型并加载 adapter
    base_model = model._first_module().auto_model
    model._first_module().auto_model = PeftModel.from_pretrained(base_model, adapter_path)
    
    return model

class OursEmbedder:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = {}

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        # Use convert_to_tensor=True for util.cos_sim
        emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        self.cache[text] = emb
        return emb

def _process_evaluations_pc(data, embedder, task_prompt: str):
    results = []
    
    for ex in tqdm(data, desc="Processing PC"):
        anchor_text = get_qwen3_query_prompt(task_prompt, get_paper_text(ex.get("anchor", {})))
        anchor_emb = embedder.encode(anchor_text)

        # Prepare base item structure
        anchor = ex.get("anchor", {})
        positive = ex.get("positive", {})
        negative = ex.get("negative", {})
        
        item = {
            "anchor_id": _get_id(anchor),
            "positive_id": _get_id(positive),
            "negative_id": _get_id(negative),
            "type": ex.get("type", "paper_centric"),
        }
        
        # Calculate scores
        pos_text = build_profile_doc_text(positive.get("papers", []) if isinstance(positive, dict) else [])
        pos_emb = embedder.encode(pos_text)
        score_pos = util.cos_sim(anchor_emb, pos_emb).item()

        neg_text = build_profile_doc_text(negative.get("papers", []) if isinstance(negative, dict) else [])
        neg_emb = embedder.encode(neg_text)
        score_neg = util.cos_sim(anchor_emb, neg_emb).item()

        item["positive_score"] = score_pos
        item["negative_score"] = score_neg
        item["positive_ref_score"] = positive.get("score")
        item["negative_ref_score"] = negative.get("score")
        results.append(item)
            
    return results

def _process_evaluations_rc(data, embedder, task_prompt_rc: str):
    results = []
    
    for ex in tqdm(data, desc="Processing RC"):
        anchor, anchor_pos_source, anchor_neg_source = _resolve_rc_anchor_profiles(ex)
        positive = ex.get("positive", {})
        negative = ex.get("negative", {})
        anchor_for_pos = anchor_pos_source
        anchor_for_neg = anchor_neg_source
        anchor_text_pos = get_qwen3_query_prompt(
            task_prompt_rc,
            build_profile_text(anchor_for_pos.get("papers", []) if isinstance(anchor_for_pos, dict) else []),
        )
        anchor_emb_pos = embedder.encode(anchor_text_pos)
        anchor_text_neg = get_qwen3_query_prompt(
            task_prompt_rc,
            build_profile_text(anchor_for_neg.get("papers", []) if isinstance(anchor_for_neg, dict) else []),
        )
        anchor_emb_neg = embedder.encode(anchor_text_neg)

        item = {
            "anchor_id": _get_id(anchor),
            "positive_id": _get_id(positive),
            "negative_id": _get_id(negative),
            "type": ex.get("type", "reviewer_centric"),
        }

        # Positive (Paper)
        pos_text = get_paper_text(positive)
        pos_emb = embedder.encode(pos_text)
        score_pos = util.cos_sim(anchor_emb_pos, pos_emb).item()

        # Negative (Paper)
        neg_text = get_paper_text(negative)
        neg_emb = embedder.encode(neg_text)
        score_neg = util.cos_sim(anchor_emb_neg, neg_emb).item()

        item["positive_score"] = score_pos
        item["negative_score"] = score_neg
        item["positive_ref_score"] = positive.get("score")
        item["negative_ref_score"] = negative.get("score")
        results.append(item)

    return results

def calculate_similarity(
    model_path: str,
    data_path: str,
    output_path: str,
    device: str,
    adapter_path: str = None,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    task_prompt_rc: str = DEFAULT_TASK_PROMPT_RC,
):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("data_path must be a JSON list")

    dataset_type = _detect_dataset_type(data)

    print(f"Detected dataset type: {dataset_type}")

    print(f"Loading model from {model_path}...")
    
    if adapter_path:
        embedder_model = load_st_model_with_adapter(model_path, adapter_path, device)
    else:
        # Resolve absolute path to ensure local loading
        abs_model_path = os.path.abspath(model_path)
        print(f"Loading local model from {abs_model_path}...")
        embedder_model = SentenceTransformer(abs_model_path, device=device, trust_remote_code=True)
        
    embedder = OursEmbedder(embedder_model, device)

    if dataset_type == "paper_centric":
        results = _process_evaluations_pc(data, embedder, task_prompt=task_prompt)
    elif dataset_type == "reviewer_centric":
        results = _process_evaluations_rc(data, embedder, task_prompt_rc=task_prompt_rc)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type!r}. Expect 'paper_centric' or 'reviewer_centric'.")

    if output_path:
        base_output_dir = os.path.dirname(output_path) or "."
        _ensure_dir(base_output_dir)
        print(f"Saving results to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def evaluate_flat_results(results: list) -> dict:
    nested_data = []
    for item in results:
        nested_data.append(
            {
                "positive": {"pred_score": item.get("positive_score"), "score": item.get("positive_ref_score")},
                "negative": {"pred_score": item.get("negative_score"), "score": item.get("negative_ref_score")},
            }
        )
    res_loss = compute_pairwise_weighted_kendall_loss(
        nested_data,
        positive_key="positive",
        negative_key="negative",
        pred_field="pred_score",
        ref_field="score",
    )
    res_acc = compute_pairwise_accuracy(
        nested_data,
        positive_key="positive",
        negative_key="negative",
        pred_field="pred_score",
        ref_field="score",
    )
    return {
        "loss": float(res_loss["loss"]),
        "acc": float(res_acc["accuracy"]),
        "used": int(res_loss.get("used", 0)),
        "skipped": int(max(res_loss.get("skipped", 0), res_acc.get("skipped", 0))),
        "total": int(len(results)),
    }

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to PEFT adapter (optional)")
    parser.add_argument("--data_path", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0'). Default is 'auto'.")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--task_prompt", type=str, default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--task_prompt_rc", type=str, default=DEFAULT_TASK_PROMPT_RC)
    return parser.parse_args()

def _load_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config_path)

    model_path = args.model_path or config.get("model_path")
    if not model_path:
        raise ValueError("Please provide --model_path or set it in config.")
    
    adapter_path = args.adapter_path or config.get("adapter_path")
    task_prompt = args.task_prompt if args.task_prompt is not None else config.get("task_prompt", DEFAULT_TASK_PROMPT)
    task_prompt_rc = args.task_prompt_rc if args.task_prompt_rc is not None else config.get("task_prompt_rc", DEFAULT_TASK_PROMPT_RC)
    
    data_paths = args.data_path
    output_path = args.output_path
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if len(data_paths) == 1:
        if args.no_save:
            final_output_path = ""
        elif output_path.endswith(".json"):
            final_output_path = output_path
        elif output_path:
            _ensure_dir(output_path)
            final_output_path = os.path.join(output_path, f"{_output_base_for_input(data_paths[0])}.json")
        else:
            final_output_path = ""
            
        results = calculate_similarity(
            model_path,
            data_paths[0],
            final_output_path,
            device,
            adapter_path,
            task_prompt=task_prompt,
            task_prompt_rc=task_prompt_rc,
        )
        metrics = evaluate_flat_results(results)
        print(
            f"{_output_base_for_input(data_paths[0])}: "
            f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
            f"used={metrics['used']} skipped={metrics['skipped']} total={metrics['total']}"
        )
    else:
        for dp in data_paths:
            out_file = ""
            if (not args.no_save) and output_path:
                _ensure_dir(output_path)
                out_file = os.path.join(output_path, f"{_output_base_for_input(dp)}.json")
            results = calculate_similarity(
                model_path,
                dp,
                out_file,
                device,
                adapter_path,
                task_prompt=task_prompt,
                task_prompt_rc=task_prompt_rc,
            )
            metrics = evaluate_flat_results(results)
            print(
                f"{_output_base_for_input(dp)}: "
                f"loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
                f"used={metrics['used']} skipped={metrics['skipped']} total={metrics['total']}"
            )
