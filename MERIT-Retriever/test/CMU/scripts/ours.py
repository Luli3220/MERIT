import os
import json
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from helpers import to_dicts
from scoring import compute_kendall_stats

DEFAULT_AUTHOR_TASK = "For the Author (Query): Represent this author's publication history and expertise for finding relevant academic papers."
DEFAULT_PAPER_TASK = "For the Paper (Document): Represent this academic paper's title and abstract for research recommendation and retrieval."


def build_paper_text(item: dict) -> str:
    content = item.get("content", {}) if isinstance(item, dict) else {}
    title = (content.get("title") or item.get("title") or "").strip()
    abstract = (content.get("abstract") or item.get("abstract") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts)


def load_reviewer_papers(reviewer_file: str) -> list[dict]:
    papers = []
    with open(reviewer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                papers.append(obj)
    return papers


def build_author_history_text_from_papers(papers: list[dict]) -> str:
    chunks = []
    for idx, paper in enumerate(papers, start=1):
        text = build_paper_text(paper)
        if text:
            chunks.append(f"Paper {idx}:\n{text}")
    return "\n\n".join(chunks)


def build_author_history_text(reviewer_file: str) -> str:
    papers = load_reviewer_papers(reviewer_file)
    return build_author_history_text_from_papers(papers)


def to_query(task: str, text: str) -> str:
    return f"Instruct: {task}\nQuery: {text}"


def load_model(model_path: str, adapter_path: str | None, device: str) -> SentenceTransformer:
    if adapter_path:
        from peft import PeftModel
        model = SentenceTransformer(os.path.abspath(model_path), trust_remote_code=True, device=device)
        base_model = model._first_module().auto_model
        model._first_module().auto_model = PeftModel.from_pretrained(base_model, os.path.abspath(adapter_path))
        return model
    return SentenceTransformer(os.path.abspath(model_path), trust_remote_code=True, device=device)


def process_dataset(
    dataset_path: str,
    model: SentenceTransformer,
    author_task: str,
    target_papers: set[str],
    target_reviewers: set[str],
    batch_size: int,
    topk_path: str,
    save_path: str = "",
) -> dict:
    submissions_file = os.path.join(dataset_path, "submissions.json")
    with open(submissions_file, "r", encoding="utf-8") as f:
        submissions = json.load(f)
    with open(topk_path, "r", encoding="utf-8") as f:
        topk_map = json.load(f)

    paper_ids = [pid for pid in submissions.keys() if pid in target_papers]
    paper_texts = [build_paper_text(submissions[pid]) for pid in paper_ids]
    paper_embeddings = model.encode(
        paper_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    archives_dir = os.path.join(dataset_path, "archives")
    reviewer_ids = []
    reviewer_files = []
    for fname in sorted(os.listdir(archives_dir)):
        if not fname.endswith(".jsonl"):
            continue
        rid = fname.replace(".jsonl", "").replace("~", "")
        if rid not in target_reviewers:
            continue
        reviewer_ids.append(rid)
        reviewer_files.append(os.path.join(archives_dir, fname))

    preds = {}
    for rid, reviewer_file in tqdm(list(zip(reviewer_ids, reviewer_files)), desc=f"{os.path.basename(dataset_path)} reviewers"):
        preds[rid] = {}
        reviewer_papers = load_reviewer_papers(reviewer_file)
        reviewer_paper_map = {str(p.get("id", "")): p for p in reviewer_papers if isinstance(p, dict)}
        reviewer_topk = topk_map.get(rid, {})
        if len(reviewer_paper_map) == 0:
            empty_profile = build_author_history_text_from_papers([])
            reviewer_text = to_query(author_task, empty_profile)
            reviewer_embedding = model.encode(
                reviewer_text,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            reviewer_embedding = np.asarray(reviewer_embedding).reshape(-1)
            scores = np.dot(paper_embeddings, reviewer_embedding)
            for p_idx, pid in enumerate(paper_ids):
                preds[rid][pid] = float(scores[p_idx])
            continue

        reviewer_emb_cache: dict[tuple[int, ...], np.ndarray] = {}
        for p_idx, pid in enumerate(paper_ids):
            selected_ids = [paper_id for paper_id in reviewer_topk.get(pid, []) if paper_id in reviewer_paper_map]
            cache_key = tuple(selected_ids)
            if cache_key not in reviewer_emb_cache:
                selected = [reviewer_paper_map[paper_id] for paper_id in selected_ids]
                profile = build_author_history_text_from_papers(selected)
                reviewer_text = to_query(author_task, profile)
                reviewer_embedding = model.encode(
                    reviewer_text,
                    batch_size=1,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                reviewer_emb_cache[cache_key] = np.asarray(reviewer_embedding).reshape(-1)
            preds[rid][pid] = float(np.dot(paper_embeddings[p_idx], reviewer_emb_cache[cache_key]))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)
    return preds


def evaluate_predictions(preds: dict, refs: dict) -> dict:
    all_reviewers = list(refs.keys())
    all_papers = set()
    for rev in refs:
        all_papers = all_papers.union(refs[rev].keys())
    return compute_kendall_stats(preds, refs, all_papers, all_reviewers)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base_dir", type=str, default=os.path.join(base_dir, "evaluation_datasets"))
    parser.add_argument("--dataset_csv", type=str, default=os.path.join(base_dir, "data", "evaluations.csv"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--author_task", type=str, default=DEFAULT_AUTHOR_TASK)
    parser.add_argument("--save_predictions_dir", type=str, default="")
    parser.add_argument("--topk_dir", type=str, required=True)
    parser.add_argument("--prediction_prefix", type=str, default="ours_rc")
    parser.add_argument("--profile_top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = load_model(args.model_path, args.adapter_path, args.device)
    df = pd.read_csv(args.dataset_csv, sep="\t")
    refs, _ = to_dicts(df)
    target_reviewers = set(refs.keys())
    target_papers = set()
    for rid in refs:
        target_papers.update(refs[rid].keys())

    losses = []
    accs = []
    for i in range(args.start, args.end + 1):
        dataset_name = f"d_20_{i}"
        dataset_path = os.path.join(args.base_dir, dataset_name)
        if not os.path.exists(dataset_path):
            continue
        topk_path = os.path.join(args.topk_dir, f"{args.prediction_prefix}_{dataset_name}_topk{int(args.profile_top_k)}.json")
        if not os.path.exists(topk_path):
            raise FileNotFoundError(topk_path)
        out_file = ""
        if args.save_predictions_dir:
            out_file = os.path.join(args.save_predictions_dir, f"{args.prediction_prefix}_{dataset_name}_ta.json")
        preds = process_dataset(
            dataset_path=dataset_path,
            model=model,
            author_task=args.author_task,
            target_papers=target_papers,
            target_reviewers=target_reviewers,
            batch_size=args.batch_size,
            topk_path=topk_path,
            save_path=out_file,
        )
        stats = evaluate_predictions(preds, refs)
        losses.append(float(stats["loss"]))
        accs.append(float(stats["acc"]))
        print(f"{dataset_name}: loss={stats['loss']:.4f} acc={stats['acc']:.4f}")

    if losses:
        print(f"Pointwise estimate of loss: {float(np.mean(losses)):.4f}")
        print(f"Pointwise estimate of acc: {float(np.mean(accs)):.4f}")
    else:
        print("No datasets were evaluated.")


if __name__ == "__main__":
    main()
