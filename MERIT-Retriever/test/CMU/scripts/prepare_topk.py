import argparse
import json
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from helpers import to_dicts

PAPER_INSTRUCTION = (
    "For the Paper (Document):\n"
    "Represent this academic paper's title and abstract for research recommendation and retrieval:"
)


def build_paper_text(item: dict) -> str:
    content = item.get("content", {}) if isinstance(item, dict) else {}
    title = (content.get("title") or item.get("title") or "").strip()
    abstract = (content.get("abstract") or item.get("abstract") or "").strip()
    if title:
        title_text = f"Title: {title}"
        if abstract:
            return f"{PAPER_INSTRUCTION}\n{title_text}\nAbstract: {abstract}"
        return f"{PAPER_INSTRUCTION}\n{title_text}"
    if abstract:
        return f"{PAPER_INSTRUCTION}\nAbstract: {abstract}"
    return ""


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


def load_prefilter_model(model_path: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(os.path.abspath(model_path), trust_remote_code=True, device=device)


def prepare_dataset_topk(
    dataset_path: str,
    prefilter_model: SentenceTransformer,
    target_papers: set[str],
    target_reviewers: set[str],
    batch_size: int,
    profile_top_k: int,
    save_path: str = "",
) -> dict:
    submissions_file = os.path.join(dataset_path, "submissions.json")
    with open(submissions_file, "r", encoding="utf-8") as f:
        submissions = json.load(f)

    paper_ids = [pid for pid in submissions.keys() if pid in target_papers]
    paper_texts = [build_paper_text(submissions[pid]) for pid in paper_ids]
    paper_embeddings = prefilter_model.encode(
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

    topk_results = {}
    k = int(profile_top_k)
    for rid, reviewer_file in tqdm(list(zip(reviewer_ids, reviewer_files)), desc=f"{os.path.basename(dataset_path)} reviewers"):
        reviewer_papers = load_reviewer_papers(reviewer_file)
        candidate_pairs = []
        for p in reviewer_papers:
            txt = build_paper_text(p)
            if txt:
                candidate_pairs.append((p, txt))
        topk_results[rid] = {}
        if len(candidate_pairs) == 0:
            for pid in paper_ids:
                topk_results[rid][pid] = []
            continue

        candidate_papers = [x[0] for x in candidate_pairs]
        candidate_texts = [x[1] for x in candidate_pairs]
        candidate_ids = [str(p.get("id", "")) for p in candidate_papers]

        if len(candidate_papers) <= k:
            for pid in paper_ids:
                topk_results[rid][pid] = candidate_ids
            continue

        candidate_embeddings = prefilter_model.encode(
            candidate_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for p_idx, pid in enumerate(paper_ids):
            query_emb = np.asarray(paper_embeddings[p_idx]).reshape(-1)
            sims = np.dot(candidate_embeddings, query_emb)
            top_idx = np.argsort(-sims)[:k]
            topk_results[rid][pid] = [candidate_ids[int(i)] for i in top_idx.tolist()]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(topk_results, f)
    return topk_results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--base_dir", type=str, default=os.path.join(base_dir, "evaluation_datasets"))
    parser.add_argument("--dataset_csv", type=str, default=os.path.join(base_dir, "data", "evaluations.csv"))
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--profile_top_k", type=int, default=3)
    parser.add_argument("--save_topk_dir", type=str, default="")
    parser.add_argument("--topk_prefix", type=str, default="ours_rc")
    args = parser.parse_args()

    prefilter_model = load_prefilter_model(args.model_path, args.device)
    df = pd.read_csv(args.dataset_csv, sep="\t")
    refs, _ = to_dicts(df)
    target_reviewers = set(refs.keys())
    target_papers = set()
    for rid in refs:
        target_papers.update(refs[rid].keys())

    for i in range(args.start, args.end + 1):
        dataset_name = f"d_20_{i}"
        dataset_path = os.path.join(args.base_dir, dataset_name)
        if not os.path.exists(dataset_path):
            continue
        out_file = ""
        if args.save_topk_dir:
            out_file = os.path.join(args.save_topk_dir, f"{args.topk_prefix}_{dataset_name}_topk{int(args.profile_top_k)}.json")
        prepare_dataset_topk(
            dataset_path=dataset_path,
            prefilter_model=prefilter_model,
            target_papers=target_papers,
            target_reviewers=target_reviewers,
            batch_size=args.batch_size,
            profile_top_k=args.profile_top_k,
            save_path=out_file,
        )
        print(f"{dataset_name}: top-k prepared")


if __name__ == "__main__":
    main()
