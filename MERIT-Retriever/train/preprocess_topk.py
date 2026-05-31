#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

PAPER_INSTRUCTION = (
    "For the Paper (Document):\n"
    "Represent this academic paper's title and abstract for research recommendation and retrieval:"
)


def load_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 不是 JSON 数组")
    return data


def save_json_array(path: Path, data: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def paper_text(p: Dict[str, Any]) -> str:
    title = (p.get("title") or p.get("paper_title") or "").strip()
    abstract = (p.get("abstract") or "").strip()
    if title and abstract:
        body = f"Title: {title}\nAbstract: {abstract}"
        return f"{PAPER_INSTRUCTION}\n{body}"
    if title:
        body = f"Title: {title}"
        return f"{PAPER_INSTRUCTION}\n{body}"
    if abstract:
        body = f"Abstract: {abstract}"
        return f"{PAPER_INSTRUCTION}\n{body}"
    return ""


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    bsz = last_hidden_states.size(0)
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), lengths]


class Encoder:
    def __init__(self, model_name: str, max_length: int, local_files_only: bool, device: str):
        self.device = torch.device(device)
        self.max_length = max_length
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only, padding_side="right")
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only,torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        self.dim = int(getattr(self.model.config, "hidden_size", 4096))
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode_many(self, texts: List[str], batch_size: int = 32) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = [torch.zeros(self.dim, dtype=torch.float32) for _ in texts]
        uncached_positions: List[int] = []
        uncached_texts: List[str] = []
        for i, t in enumerate(texts):
            key = t.strip()
            if not key:
                outputs[i] = torch.zeros(self.dim, dtype=torch.float32)
                continue
            if key in self.cache:
                outputs[i] = self.cache[key]
                continue
            uncached_positions.append(i)
            uncached_texts.append(key)
        if len(uncached_texts) == 0:
            return outputs
        bs = max(1, int(batch_size))
        for start in range(0, len(uncached_texts), bs):
            chunk_texts = uncached_texts[start:start + bs]
            toks = self.tok(
                chunk_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            emb = last_token_pool(out.last_hidden_state, toks["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1).detach().cpu().float()
            for j in range(emb.size(0)):
                text_key = chunk_texts[j]
                pos = uncached_positions[start + j]
                self.cache[text_key] = emb[j]
                outputs[pos] = emb[j]
        return outputs

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        return self.encode_many([text], batch_size=1)[0]


def topk_papers_by_query(
    papers: List[Dict[str, Any]],
    query_text: str,
    top_k: int,
    encoder: Encoder,
    batch_size: int,
) -> List[Dict[str, Any]]:
    if not isinstance(papers, list) or len(papers) <= top_k:
        return papers
    if not isinstance(query_text, str) or not query_text.strip():
        return papers[:top_k]
    q = encoder.encode(query_text)
    cand_texts = [paper_text(p if isinstance(p, dict) else {}) for p in papers]
    cand_embs = encoder.encode_many(cand_texts, batch_size=batch_size)
    scored = []
    for p, d in zip(papers, cand_embs):
        s = torch.dot(q, d).item()
        scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def row_target_text_from_side(block: Dict[str, Any]) -> str:
    if not isinstance(block, dict):
        return ""
    papers = block.get("papers", [])
    first = papers[0] if isinstance(papers, list) and len(papers) > 0 and isinstance(papers[0], dict) else {}
    return paper_text(
        {
            "title": block.get("title", block.get("paper_title", first.get("title", first.get("paper_title", "")))),
            "abstract": block.get("abstract", first.get("abstract", "")),
        }
    )


def merge_unique_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    merged = []
    for p in papers:
        if not isinstance(p, dict):
            continue
        key = (
            (p.get("paper_id") or "").strip(),
            (p.get("title") or p.get("paper_title") or "").strip(),
            (p.get("abstract") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return merged


def process_rows(rows: List[Dict[str, Any]], top_k: int, encoder: Encoder, batch_size: int, desc: str) -> List[Dict[str, Any]]:
    out = []
    for row in tqdm(rows, desc=desc, dynamic_ncols=True):
        row = dict(row)
        ex_type = row.get("type", "paper_centric")
        anchor = row.get("anchor", {}) if isinstance(row.get("anchor", {}), dict) else {}
        if ex_type == "paper_centric":
            query_text = paper_text(
                {
                    "title": anchor.get("paper_title", anchor.get("title", row.get("title", ""))),
                    "abstract": anchor.get("abstract", row.get("abstract", "")),
                }
            )
            for side in ("positive", "negative"):
                block = row.get(side, {})
                if not isinstance(block, dict):
                    continue
                papers = block.get("papers", [])
                if isinstance(papers, list) and len(papers) > 0:
                    block["papers"] = topk_papers_by_query(papers, query_text, top_k, encoder, batch_size)
                    row[side] = block
        elif ex_type == "reviewer_centric":
            anchor_papers = anchor.get("papers", [])
            if isinstance(anchor_papers, list) and len(anchor_papers) > 0:
                pos = row.get("positive", {}) if isinstance(row.get("positive", {}), dict) else {}
                neg = row.get("negative", {}) if isinstance(row.get("negative", {}), dict) else {}
                pos_query_text = row_target_text_from_side(pos)
                neg_query_text = row_target_text_from_side(neg)
                pos_profile = topk_papers_by_query(anchor_papers, pos_query_text, top_k, encoder, batch_size)
                neg_profile = topk_papers_by_query(anchor_papers, neg_query_text, top_k, encoder, batch_size)
                row["anchor_for_positive"] = {"papers": pos_profile}
                row["anchor_for_negative"] = {"papers": neg_profile}
                merged_profile = merge_unique_papers(pos_profile + neg_profile)
                combined_query_text = "\n".join([t for t in [pos_query_text, neg_query_text] if isinstance(t, str) and t.strip()])
                anchor["papers"] = topk_papers_by_query(merged_profile, combined_query_text, top_k, encoder, batch_size)
                row["anchor"] = anchor
        out.append(row)
    return out


def derive_output_path(path: Path, suffix: str) -> Path:
    stem = path.stem
    return path.with_name(f"{stem}_{suffix}{path.suffix}")


def main():
    train_dir = Path(__file__).resolve().parent
    retriever_dir = train_dir.parent
    default_raw_dir = train_dir / "data" / "raw"
    default_processed_dir = train_dir / "data" / "processed"
    default_model_name = os.environ.get("MODEL_NAME", str(retriever_dir / "Models" / "Qwen3-Embedding-8B"))
    parser = argparse.ArgumentParser(description="使用backbone按相似度抽取作者画像Top-K（默认K=3）")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--pc_file", type=Path, default=default_raw_dir / "train_pc.json", help=argparse.SUPPRESS)
    parser.add_argument("--rc_file", type=Path, default=default_raw_dir / "train_rc.json", help=argparse.SUPPRESS)
    parser.add_argument("--pc_val_file", type=Path, default=default_raw_dir / "RATE_pc_val.json", help=argparse.SUPPRESS)
    parser.add_argument("--rc_val_file", type=Path, default=default_raw_dir / "RATE_rc_val.json", help=argparse.SUPPRESS)
    parser.add_argument("--output_train_pc", type=Path, default=default_processed_dir / "train_pc.json", help=argparse.SUPPRESS)
    parser.add_argument("--output_val_pc", type=Path, default=default_processed_dir / "val_pc.json", help=argparse.SUPPRESS)
    parser.add_argument("--output_train_rc", type=Path, default=default_processed_dir / "train_rc.json", help=argparse.SUPPRESS)
    parser.add_argument("--output_val_rc", type=Path, default=default_processed_dir / "val_rc.json", help=argparse.SUPPRESS)
    parser.add_argument("--model_name", type=str, default=default_model_name, help=argparse.SUPPRESS)
    parser.add_argument("--max_length", type=int, default=2048, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="cuda:2", help=argparse.SUPPRESS)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", default=True, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("top_k 必须 >= 1")
    if args.batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")

    pc_file = args.pc_file.resolve()
    rc_file = args.rc_file.resolve()
    if not pc_file.exists():
        raise FileNotFoundError(f"找不到文件: {pc_file}")
    if not rc_file.exists():
        raise FileNotFoundError(f"找不到文件: {rc_file}")
    pc_val_file = args.pc_val_file.resolve()
    rc_val_file = args.rc_val_file.resolve()
    if not pc_val_file.exists():
        raise FileNotFoundError(f"找不到 PC 验证集文件: {pc_val_file}")
    if not rc_val_file.exists():
        raise FileNotFoundError(f"找不到 RC 验证集文件: {rc_val_file}")

    output_train_pc = args.output_train_pc.resolve()
    output_val_pc = args.output_val_pc.resolve()
    output_train_rc = args.output_train_rc.resolve()
    output_val_rc = args.output_val_rc.resolve()
    topk_tag = f"topk{args.top_k}"
    if output_train_pc.name.endswith("topk5.json"):
        output_train_pc = output_train_pc.with_name(output_train_pc.name.replace("topk5.json", f"{topk_tag}.json"))
    if output_val_pc.name.endswith("topk5.json"):
        output_val_pc = output_val_pc.with_name(output_val_pc.name.replace("topk5.json", f"{topk_tag}.json"))
    if output_train_rc.name.endswith("topk5.json"):
        output_train_rc = output_train_rc.with_name(output_train_rc.name.replace("topk5.json", f"{topk_tag}.json"))
    if output_val_rc.name.endswith("topk5.json"):
        output_val_rc = output_val_rc.with_name(output_val_rc.name.replace("topk5.json", f"{topk_tag}.json"))

    outputs = {
        "train_pc": output_train_pc,
        "val_pc": output_val_pc,
        "train_rc": output_train_rc,
        "val_rc": output_val_rc,
    }
    if not args.overwrite:
        exists = [str(p) for p in outputs.values() if p.exists()]
        if exists:
            raise FileExistsError("以下文件已存在，避免覆盖请先删除或加 --overwrite:\n" + "\n".join(exists))

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    encoder = Encoder(
        model_name=args.model_name,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
        device=device,
    )

    pc_train_rows = load_json_array(pc_file)
    rc_train_rows = load_json_array(rc_file)
    pc_val_rows = load_json_array(pc_val_file)
    rc_val_rows = load_json_array(rc_val_file)

    train_pc_rows = process_rows(pc_train_rows, args.top_k, encoder, args.batch_size, desc="PC train top-k")
    val_pc_rows = process_rows(pc_val_rows, args.top_k, encoder, args.batch_size, desc="PC val top-k")
    train_rc_rows = process_rows(rc_train_rows, args.top_k, encoder, args.batch_size, desc="RC train top-k")
    val_rc_rows = process_rows(rc_val_rows, args.top_k, encoder, args.batch_size, desc="RC val top-k")

    save_json_array(outputs["train_pc"], train_pc_rows)
    save_json_array(outputs["val_pc"], val_pc_rows)
    save_json_array(outputs["train_rc"], train_rc_rows)
    save_json_array(outputs["val_rc"], val_rc_rows)

    print(f"Top-K={args.top_k}, batch_size={args.batch_size}, device={device}")
    print(f"PC source: train={pc_file} val={pc_val_file}")
    print(f"RC source: train={rc_file} val={rc_val_file}")
    print(f"train_pc: {outputs['train_pc']}")
    print(f"val_pc: {outputs['val_pc']}")
    print(f"train_rc: {outputs['train_rc']}")
    print(f"val_rc: {outputs['val_rc']}")


if __name__ == "__main__":
    main()
