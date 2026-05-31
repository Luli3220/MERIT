import argparse
import json
import os
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

from eval_LR import _ensure_dir, _output_base_for_input

PAPER_INSTRUCTION = (
    "For the Paper (Document):\n"
    "Represent this academic paper's title and abstract for research recommendation and retrieval:"
)


def _topk_base_for_input(data_path: str) -> str:
    return f"{_output_base_for_input(data_path)}_topk"


def get_paper_text(paper_data):
    title = (paper_data.get("paper_title") or paper_data.get("title") or "").strip()
    abstract = (paper_data.get("abstract") or "").strip()
    if title:
        title_text = f"Title: {title}"
        if abstract:
            return f"{PAPER_INSTRUCTION}\n{title_text}\nAbstract: {abstract}"
        return f"{PAPER_INSTRUCTION}\n{title_text}"
    if abstract:
        return f"{PAPER_INSTRUCTION}\nAbstract: {abstract}"
    return ""


def build_prefilter_paper_text(paper_data: dict) -> str:
    if not isinstance(paper_data, dict):
        return ""
    papers = paper_data.get("papers", [])
    first = papers[0] if isinstance(papers, list) and len(papers) > 0 and isinstance(papers[0], dict) else {}
    return get_paper_text(
        {
            "title": paper_data.get("title", paper_data.get("paper_title", first.get("title", first.get("paper_title", "")))),
            "abstract": paper_data.get("abstract", first.get("abstract", "")),
        }
    )


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    bsz = last_hidden_states.size(0)
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), lengths]


class BackbonePaperEncoder:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        local_files_only: bool = True,
        max_length: int = 2048,
        batch_size: int = 64,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only, padding_side="right")
        model_kwargs = {"local_files_only": local_files_only}
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs).to(self.device)
        self.model.eval()
        self.cache = {}

    @torch.no_grad()
    def encode_many(self, texts: list[str]) -> list[torch.Tensor]:
        cleaned = [text.strip() for text in texts]
        results = [None] * len(cleaned)
        pending_pairs = [(idx, text) for idx, text in enumerate(cleaned) if text not in self.cache]
        for idx, text in enumerate(cleaned):
            if text in self.cache:
                results[idx] = self.cache[text]
        for start in range(0, len(pending_pairs), self.batch_size):
            batch_pairs = pending_pairs[start : start + self.batch_size]
            batch_texts = [text for _, text in batch_pairs]
            toks = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            embs = last_token_pool(out.last_hidden_state, toks["attention_mask"])
            embs = F.normalize(embs, p=2, dim=1).detach().cpu().float()
            for (idx, text), emb in zip(batch_pairs, embs):
                self.cache[text] = emb
                results[idx] = emb
        return results

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        return self.encode_many([text])[0]


def prefilter_reviewer_profile(reviewer_data: dict, target_paper_data: dict, encoder: BackbonePaperEncoder, top_k: int) -> dict:
    if encoder is None:
        return reviewer_data
    if not isinstance(reviewer_data, dict):
        return reviewer_data
    papers = reviewer_data.get("papers", [])
    if not isinstance(papers, list) or len(papers) <= int(top_k):
        return reviewer_data
    target_text = build_prefilter_paper_text(target_paper_data)
    if not target_text.strip():
        return reviewer_data
    target_emb = encoder.encode(target_text)
    paper_texts = [build_prefilter_paper_text(p) for p in papers]
    paper_embs = encoder.encode_many(paper_texts)
    scored = [(torch.dot(target_emb, p_emb).item(), p) for p, p_emb in zip(papers, paper_embs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [x[1] for x in scored[: int(top_k)]]
    filtered = dict(reviewer_data)
    filtered["papers"] = keep
    return filtered


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
    if not dataset_type and data and "anchor" in data[0]:
        anchor = data[0]["anchor"]
        dataset_type = "reviewer_centric" if "papers" in anchor else "paper_centric"
    return dataset_type


def _prepare_with_topk_profiles(data: list, dataset_type: str, profile_top_k: int, prefilter_encoder: BackbonePaperEncoder) -> list:
    prepared = []
    for ex in data:
        if not isinstance(ex, dict):
            prepared.append(ex)
            continue
        row = dict(ex)
        if dataset_type == "paper_centric":
            anchor = row.get("anchor", {}) if isinstance(row.get("anchor", {}), dict) else {}
            positive = row.get("positive", {}) if isinstance(row.get("positive", {}), dict) else {}
            negative = row.get("negative", {}) if isinstance(row.get("negative", {}), dict) else {}
            row["positive"] = prefilter_reviewer_profile(positive, anchor, prefilter_encoder, profile_top_k)
            row["negative"] = prefilter_reviewer_profile(negative, anchor, prefilter_encoder, profile_top_k)
        elif dataset_type == "reviewer_centric":
            _, anchor_pos_source, anchor_neg_source = _resolve_rc_anchor_profiles(row)
            positive = row.get("positive", {}) if isinstance(row.get("positive", {}), dict) else {}
            negative = row.get("negative", {}) if isinstance(row.get("negative", {}), dict) else {}
            row["anchor_for_positive"] = prefilter_reviewer_profile(anchor_pos_source, positive, prefilter_encoder, profile_top_k)
            row["anchor_for_negative"] = prefilter_reviewer_profile(anchor_neg_source, negative, prefilter_encoder, profile_top_k)
        prepared.append(row)
    return prepared


def prepare_topk_data(
    data_path: str,
    output_path: str,
    profile_top_k: int,
    prefilter_encoder: BackbonePaperEncoder,
) -> tuple[str, int]:
    print(f"Loading data from {data_path} for top-k preparation...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("data_path must be a JSON list")
    dataset_type = _detect_dataset_type(data)
    if not dataset_type:
        raise ValueError(f"Unable to infer dataset type from {data_path}")
    prepared = _prepare_with_topk_profiles(
        data=data,
        dataset_type=dataset_type,
        profile_top_k=profile_top_k,
        prefilter_encoder=prefilter_encoder,
    )
    base_output_dir = os.path.dirname(output_path) or "."
    _ensure_dir(base_output_dir)
    print(f"Saving top-k data to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prepared, f, indent=2, ensure_ascii=False)
    return dataset_type, len(prepared)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--profile_top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    prefilter_encoder = None
    if args.profile_top_k and int(args.profile_top_k) > 0:
        prefilter_encoder = BackbonePaperEncoder(
            model_name=args.model_path,
            device=device,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

    if len(args.data_path) == 1:
        if args.output_path.endswith(".json"):
            final_output_path = args.output_path
        else:
            _ensure_dir(args.output_path)
            final_output_path = os.path.join(args.output_path, f"{_topk_base_for_input(args.data_path[0])}.json")
        dataset_type, total_rows = prepare_topk_data(
            data_path=args.data_path[0],
            output_path=final_output_path,
            profile_top_k=args.profile_top_k,
            prefilter_encoder=prefilter_encoder,
        )
        print(f"{_topk_base_for_input(args.data_path[0])}: type={dataset_type} total={total_rows}")
    else:
        _ensure_dir(args.output_path)
        for dp in args.data_path:
            final_output_path = os.path.join(args.output_path, f"{_topk_base_for_input(dp)}.json")
            dataset_type, total_rows = prepare_topk_data(
                data_path=dp,
                output_path=final_output_path,
                profile_top_k=args.profile_top_k,
                prefilter_encoder=prefilter_encoder,
            )
            print(f"{_topk_base_for_input(dp)}: type={dataset_type} total={total_rows}")
