import os
import json
import math
import time
import random
import hashlib
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm.auto import tqdm


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_fold_id(group: str, k: int) -> int:
    h = hashlib.md5(group.encode("utf-8")).hexdigest()
    return int(h, 16) % k


def build_paper_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    return "\n".join(parts)


def build_profile_text(papers_list: List[Dict[str, Any]]) -> str:
    chunks = []
    for idx, p in enumerate(papers_list, start=1):
        if not isinstance(p, dict):
            continue
        title = p.get("title", p.get("paper_title", ""))
        abstract = p.get("abstract", "")
        text = build_paper_text(title, abstract)
        if text:
            chunks.append(f"Paper {idx}:\n{text}")
    return "\n\n".join(chunks)


def build_profile_doc_text(papers_list: List[Dict[str, Any]]) -> str:
    merged = build_profile_text(papers_list)
    return f"Reviewer profile:\n{merged}" if merged else "Reviewer profile:"


def get_qwen3_query_prompt(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1) - 1
    lengths = lengths.clamp(min=0)
    bsz = last_hidden_states.size(0)
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), lengths]


class GoldDPODataset(Dataset):
    def __init__(self, path: str, fold_k: int, fold_id: int, split: str, presplit: bool = False, allowed_types: Optional[List[str]] = None):
        assert split in ("train", "val", "test")
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                all_rows = json.load(f)
            else:
                all_rows = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    all_rows.append(json.loads(line))

        def type_ok(ex: Dict[str, Any]) -> bool:
            if allowed_types is None:
                return True
            return ex.get("type") in allowed_types

        if presplit:
            self.rows = [ex for ex in all_rows if type_ok(ex)]
            return

        for idx, ex in enumerate(all_rows):
            if not type_ok(ex):
                continue
            anchor = ex.get("anchor", {}) if isinstance(ex, dict) else {}
            split_key = (
                ex.get("paper_id")
                or anchor.get("paper_id")
                or anchor.get("id")
                or ex.get("title")
                or anchor.get("paper_title")
                or anchor.get("title")
                or f"row_{idx}"
            )
            fid = stable_fold_id(str(split_key), fold_k)
            is_val = fid == fold_id
            if (split == "val" and is_val) or (split == "train" and not is_val):
                self.rows.append(ex)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.rows[idx]
        ex_type = ex.get("type", "paper_centric")
        anchor = ex.get("anchor", {})
        anchor_for_positive = ex.get("anchor_for_positive", {})
        anchor_for_negative = ex.get("anchor_for_negative", {})
        paper_id = ex.get("paper_id", anchor.get("paper_id", ""))
        paper_title = ex.get("title", anchor.get("paper_title", anchor.get("title", "")))
        paper_abs = ex.get("abstract", anchor.get("abstract", ""))
        pos = ex.get("positive", {})
        neg = ex.get("negative", {})
        pos_papers = pos.get("papers", []) if isinstance(pos, dict) else []
        neg_papers = neg.get("papers", []) if isinstance(neg, dict) else []
        pos_first = pos_papers[0] if isinstance(pos_papers, list) and len(pos_papers) > 0 else {}
        neg_first = neg_papers[0] if isinstance(neg_papers, list) and len(neg_papers) > 0 else {}
        pos_title = pos.get("paper_title", pos.get("title", pos.get("paper_id", pos_first.get("title", pos_first.get("paper_title", pos_first.get("paper_id", ""))))))
        pos_abs = pos.get("abstract", pos_first.get("abstract", ""))
        neg_title = neg.get("paper_title", neg.get("title", neg.get("paper_id", neg_first.get("title", neg_first.get("paper_title", neg_first.get("paper_id", ""))))))
        neg_abs = neg.get("abstract", neg_first.get("abstract", ""))
        return {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "paper_abstract": paper_abs,
            "anchor_papers": anchor.get("papers", []),
            "anchor_papers_pos": anchor_for_positive.get("papers", []) if isinstance(anchor_for_positive, dict) else [],
            "anchor_papers_neg": anchor_for_negative.get("papers", []) if isinstance(anchor_for_negative, dict) else [],
            "pos_title": pos_title,
            "pos_abstract": pos_abs,
            "neg_title": neg_title,
            "neg_abstract": neg_abs,
            "pos_papers": pos_papers if isinstance(pos_papers, list) else [],
            "neg_papers": neg_papers if isinstance(neg_papers, list) else [],
            "pos_score": pos.get("score", None) if isinstance(pos, dict) else None,
            "neg_score": neg.get("score", None) if isinstance(neg, dict) else None,
            "type": ex_type,
        }


@dataclass
class Batch:
    q_pos_input: Dict[str, torch.Tensor]
    q_neg_input: Dict[str, torch.Tensor]
    pos_input: Dict[str, torch.Tensor]
    neg_input: Dict[str, torch.Tensor]
    pair_sample_weight: torch.Tensor
    pair_pos_score: torch.Tensor
    pair_neg_score: torch.Tensor
    pair_has_ref: torch.Tensor


class PairCollator:
    def __init__(self, tokenizer: AutoTokenizer, task_prompt: str, task_prompt_rc: str, max_len_q: int, max_len_d: int):
        self.tok = tokenizer
        self.task_prompt = task_prompt
        self.task_prompt_rc = task_prompt_rc
        self.max_len_q = max_len_q
        self.max_len_d = max_len_d

    def __call__(self, batch: List[Dict[str, Any]]) -> Batch:
        q_pos_texts, q_neg_texts, pos_texts, neg_texts = [], [], [], []
        pair_weights = []
        pair_pos_scores = []
        pair_neg_scores = []
        pair_has_ref = []
        for ex in batch:
            pos_score = ex.get("pos_score", None)
            neg_score = ex.get("neg_score", None)
            if isinstance(pos_score, (int, float)) and isinstance(neg_score, (int, float)):
                pos_score_f = float(pos_score)
                neg_score_f = float(neg_score)
                pair_weights.append(float(abs(pos_score_f - neg_score_f)))
                pair_pos_scores.append(pos_score_f)
                pair_neg_scores.append(neg_score_f)
                pair_has_ref.append(1.0)
            else:
                pair_weights.append(0.0)
                pair_pos_scores.append(0.0)
                pair_neg_scores.append(0.0)
                pair_has_ref.append(0.0)
            ex_type = ex.get("type", "paper_centric")
            if ex_type == "reviewer_centric":
                anchor_pos_papers = ex.get("anchor_papers_pos") or ex.get("anchor_papers") or []
                anchor_neg_papers = ex.get("anchor_papers_neg") or ex.get("anchor_papers") or []
                reviewer_profile_pos = build_profile_text(anchor_pos_papers)
                reviewer_profile_neg = build_profile_text(anchor_neg_papers)
                q_pos_texts.append(get_qwen3_query_prompt(self.task_prompt_rc, reviewer_profile_pos))
                q_neg_texts.append(get_qwen3_query_prompt(self.task_prompt_rc, reviewer_profile_neg))
                pos_texts.append(build_paper_text(ex.get("pos_title", ""), ex.get("pos_abstract", "")))
                neg_texts.append(build_paper_text(ex.get("neg_title", ""), ex.get("neg_abstract", "")))
            else:
                paper = build_paper_text(ex["paper_title"], ex["paper_abstract"])
                q_text = get_qwen3_query_prompt(self.task_prompt, paper)
                q_pos_texts.append(q_text)
                q_neg_texts.append(q_text)
                pos_papers = ex.get("pos_papers") or []
                neg_papers = ex.get("neg_papers") or []
                if len(pos_papers) > 0 or len(neg_papers) > 0:
                    pos_texts.append(build_profile_doc_text(pos_papers))
                    neg_texts.append(build_profile_doc_text(neg_papers))
                else:
                    pos_texts.append(build_paper_text(ex.get("pos_title", ""), ex.get("pos_abstract", "")))
                    neg_texts.append(build_paper_text(ex.get("neg_title", ""), ex.get("neg_abstract", "")))
        q_pos = self.tok(q_pos_texts, padding=True, truncation=True, max_length=self.max_len_q, return_tensors="pt")
        q_neg = self.tok(q_neg_texts, padding=True, truncation=True, max_length=self.max_len_q, return_tensors="pt")
        pos = self.tok(pos_texts, padding=True, truncation=True, max_length=self.max_len_d, return_tensors="pt")
        neg = self.tok(neg_texts, padding=True, truncation=True, max_length=self.max_len_d, return_tensors="pt")
        pw = torch.tensor(pair_weights, dtype=torch.float32)
        ppos = torch.tensor(pair_pos_scores, dtype=torch.float32)
        pneg = torch.tensor(pair_neg_scores, dtype=torch.float32)
        phas = torch.tensor(pair_has_ref, dtype=torch.float32)
        return Batch(
            q_pos_input=q_pos,
            q_neg_input=q_neg,
            pos_input=pos,
            neg_input=neg,
            pair_sample_weight=pw,
            pair_pos_score=ppos,
            pair_neg_score=pneg,
            pair_has_ref=phas,
        )


class Qwen3Embedder(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(**batch_inputs)
        emb = last_token_pool(outputs.last_hidden_state, batch_inputs["attention_mask"])
        return F.normalize(emb, p=2, dim=1)


def infer_lora_target_modules(model: nn.Module) -> List[str]:
    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    found = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        for suffix in wanted:
            if name.endswith(suffix):
                found.add(suffix)
    return [m for m in wanted if m in found] or ["q_proj", "v_proj"]


def reduce_mean(t: torch.Tensor, accelerator: Accelerator) -> torch.Tensor:
    t = accelerator.reduce(t, reduction="mean")
    return t


def local_label_offset(local_bs: int, accelerator: Accelerator, device: torch.device) -> torch.Tensor:
    bs_tensor = torch.tensor([local_bs], device=device, dtype=torch.long)
    bs_all = accelerator.gather(bs_tensor)
    start = bs_all[:accelerator.process_index].sum()
    return start + torch.arange(local_bs, device=device, dtype=torch.long)


def mixed_contrastive_loss(
    q_pos_emb: torch.Tensor,
    q_neg_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    accelerator: Accelerator,
    temperature: float,
    inbatch_weight: float,
    pair_weight: float,
    pair_margin: float,
    pair_sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dist.is_initialized():
        all_pos = torch.cat(dist_nn.all_gather(pos_emb), dim=0)
        all_neg = torch.cat(dist_nn.all_gather(neg_emb), dim=0)
    else:
        all_pos, all_neg = pos_emb, neg_emb
    all_docs = torch.cat([all_pos, all_neg], dim=0)
    logits = (q_pos_emb @ all_docs.t()) / temperature
    labels = local_label_offset(q_pos_emb.size(0), accelerator, q_pos_emb.device)
    loss_ce = F.cross_entropy(logits, labels)
    s_pos = (q_pos_emb * pos_emb).sum(dim=1)
    s_neg = (q_neg_emb * neg_emb).sum(dim=1)
    pair_loss_each = -F.logsigmoid((s_pos - s_neg) - pair_margin)
    if pair_sample_weight is None:
        loss_pair = pair_loss_each.mean()
    else:
        w = pair_sample_weight.to(pair_loss_each.device).float()
        mask = w > 0
        if mask.any():
            ww = w[mask]
            loss_pair = (pair_loss_each[mask] * ww).sum() / ww.sum().clamp_min(1e-8)
        else:
            loss_pair = pair_loss_each.new_tensor(0.0)
    loss = inbatch_weight * loss_ce + pair_weight * loss_pair
    return loss, loss_ce.detach(), loss_pair.detach()


@torch.no_grad()
def eval_loss_and_acc(
    embedder: Qwen3Embedder,
    loader: DataLoader,
    accelerator: Accelerator,
    temperature: float,
    inbatch_weight: float,
    pair_weight: float,
    pair_margin: float,
) -> tuple[float, float, float, int, int]:
    embedder.eval()
    loss_sum = torch.tensor(0.0, device=accelerator.device)
    correct_sum = torch.tensor(0.0, device=accelerator.device)
    total_sum = torch.tensor(0.0, device=accelerator.device)
    kendall_loss_sum = torch.tensor(0.0, device=accelerator.device)
    kendall_max_sum = torch.tensor(0.0, device=accelerator.device)
    kendall_used_sum = torch.tensor(0.0, device=accelerator.device)
    kendall_skipped_sum = torch.tensor(0.0, device=accelerator.device)
    for batch in loader:
        q_pos = {k: v.to(accelerator.device) for k, v in batch.q_pos_input.items()}
        q_neg = {k: v.to(accelerator.device) for k, v in batch.q_neg_input.items()}
        pos = {k: v.to(accelerator.device) for k, v in batch.pos_input.items()}
        neg = {k: v.to(accelerator.device) for k, v in batch.neg_input.items()}
        q_pos_emb = embedder(q_pos)
        q_neg_emb = embedder(q_neg)
        pos_emb = embedder(pos)
        neg_emb = embedder(neg)
        pair_w = batch.pair_sample_weight.to(accelerator.device)
        loss, _, _ = mixed_contrastive_loss(
            q_pos_emb,
            q_neg_emb,
            pos_emb,
            neg_emb,
            accelerator=accelerator,
            temperature=temperature,
            inbatch_weight=inbatch_weight,
            pair_weight=pair_weight,
            pair_margin=pair_margin,
            pair_sample_weight=pair_w,
        )
        bs = torch.tensor(float(q_pos_emb.size(0)), device=accelerator.device)
        s_pos = (q_pos_emb * pos_emb).sum(dim=1)
        s_neg = (q_neg_emb * neg_emb).sum(dim=1)
        correct = (s_pos > s_neg).float().sum()
        pred_diff = s_pos - s_neg
        true_diff = batch.pair_pos_score.to(accelerator.device) - batch.pair_neg_score.to(accelerator.device)
        has_ref = batch.pair_has_ref.to(accelerator.device) > 0
        w = true_diff.abs()
        valid = has_ref & (w > 0)
        used = valid.float().sum()
        skipped = bs - used
        prod = pred_diff * true_diff
        tie_mask = valid & (prod == 0)
        wrong_mask = valid & (prod < 0)
        kendall_loss = (w[tie_mask] * 0.5).sum() + w[wrong_mask].sum()
        kendall_max = w[valid].sum()
        loss_sum += loss * bs
        correct_sum += correct
        total_sum += bs
        kendall_loss_sum += kendall_loss
        kendall_max_sum += kendall_max
        kendall_used_sum += used
        kendall_skipped_sum += skipped
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    correct_sum = accelerator.reduce(correct_sum, reduction="sum")
    total_sum = accelerator.reduce(total_sum, reduction="sum")
    kendall_loss_sum = accelerator.reduce(kendall_loss_sum, reduction="sum")
    kendall_max_sum = accelerator.reduce(kendall_max_sum, reduction="sum")
    kendall_used_sum = accelerator.reduce(kendall_used_sum, reduction="sum")
    kendall_skipped_sum = accelerator.reduce(kendall_skipped_sum, reduction="sum")
    if total_sum.item() == 0:
        return 0.0, 0.0, float("nan"), 0, 0
    kendall = (kendall_loss_sum / kendall_max_sum).item() if kendall_max_sum.item() > 0 else float("nan")
    return (
        (loss_sum / total_sum).item(),
        (correct_sum / total_sum).item(),
        kendall,
        int(kendall_used_sum.item()),
        int(kendall_skipped_sum.item()),
    )


def write_args_txt(args, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")


def build_view_datasets(
    train_path: str,
    val_path: Optional[str],
    view_type: str,
    fold_k: int,
    fold_id: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if val_path:
        train_ds = GoldDPODataset(train_path, fold_k=fold_k, fold_id=fold_id, split="train", presplit=True, allowed_types=[view_type])
        val_ds = GoldDPODataset(val_path, fold_k=fold_k, fold_id=fold_id, split="val", presplit=True, allowed_types=[view_type])
    else:
        train_ds = GoldDPODataset(train_path, fold_k=fold_k, fold_id=fold_id, split="train", presplit=False, allowed_types=[view_type])
        val_ds = GoldDPODataset(train_path, fold_k=fold_k, fold_id=fold_id, split="val", presplit=False, allowed_types=[view_type])
        if len(train_ds) == 0 or len(val_ds) == 0:
            full_ds = GoldDPODataset(train_path, fold_k=fold_k, fold_id=fold_id, split="train", presplit=True, allowed_types=[view_type])
            n = len(full_ds)
            if n < 2:
                raise ValueError(f"{view_type} dataset has too few samples: {n}")
            idxs = list(range(n))
            rng = random.Random(seed)
            rng.shuffle(idxs)
            cut = max(1, min(n - 1, int(n * 0.9)))
            train_ds = Subset(full_ds, idxs[:cut])
            val_ds = Subset(full_ds, idxs[cut:])
    return train_ds, val_ds


def maybe_save_best(accelerator: Accelerator, embedder: Qwen3Embedder, tokenizer: AutoTokenizer, args, best_dir: str):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_embedder = accelerator.unwrap_model(embedder)
        unwrapped_embedder.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)
        write_args_txt(args, os.path.join(best_dir, "train_args.txt"))


def main():
    train_dir = os.path.dirname(os.path.abspath(__file__))
    retriever_dir = os.path.dirname(train_dir)
    default_model_name = os.path.join(retriever_dir, "Models", "Qwen3-Embedding-8B")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=default_model_name)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gold_train_pc_path", type=str, default=None)
    parser.add_argument("--gold_train_rc_path", type=str, default=None)
    parser.add_argument("--gold_val_pc_path", type=str, default=None)
    parser.add_argument("--gold_val_rc_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--task_prompt", type=str, default="For the Paper (Query): Represent this academic paper's title and abstract for retrieving suitable reviewer profiles.")
    parser.add_argument("--task_prompt_rc", type=str, default="For the Author (Query): Represent this author's publication history and expertise for finding relevant academic papers.")
    parser.add_argument("--max_len_q", type=int, default=2048)
    parser.add_argument("--max_len_d", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--inbatch_weight", type=float, default=1.0)
    parser.add_argument("--pair_weight", type=float, default=1.0)
    parser.add_argument("--pair_margin", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--fold_k", type=int, default=5)
    parser.add_argument("--fold_id", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.06)
    parser.add_argument("--run_tag", type=str, default="")
    args = parser.parse_args()

    mixed_precision = "no"
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"
    accelerator = Accelerator(gradient_accumulation_steps=max(args.grad_accum, 1), mixed_precision=mixed_precision)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only, padding_side="right")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model_kwargs = {
        "local_files_only": args.local_files_only,
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl
    try:
        base = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    except Exception as e:
        if args.attn_impl != "flash_attention_2":
            raise
        if accelerator.is_main_process:
            print(f"[Warn] flash_attention_2 unavailable, fallback to sdpa. reason={e}")
        model_kwargs["attn_implementation"] = "sdpa"
        base = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    base.config.use_cache = False
    if args.gradient_checkpointing and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    target_modules = infer_lora_target_modules(base)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base, lora_cfg)
    embedder = Qwen3Embedder(model)

    train_parts, val_parts = [], []
    val_pc_ds = None
    val_rc_ds = None
    if args.gold_train_pc_path:
        train_pc_ds, val_pc_ds = build_view_datasets(
            train_path=args.gold_train_pc_path,
            val_path=args.gold_val_pc_path,
            view_type="paper_centric",
            fold_k=args.fold_k,
            fold_id=args.fold_id,
            seed=args.seed,
        )
        train_parts.append(train_pc_ds)
        val_parts.append(val_pc_ds)
    if args.gold_train_rc_path:
        train_rc_ds, val_rc_ds = build_view_datasets(
            train_path=args.gold_train_rc_path,
            val_path=args.gold_val_rc_path,
            view_type="reviewer_centric",
            fold_k=args.fold_k,
            fold_id=args.fold_id,
            seed=args.seed,
        )
        train_parts.append(train_rc_ds)
        val_parts.append(val_rc_ds)
    if not train_parts:
        raise ValueError("Need at least one of --gold_train_pc_path or --gold_train_rc_path")

    val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    collator = PairCollator(
        tokenizer=tokenizer,
        task_prompt=args.task_prompt,
        task_prompt_rc=args.task_prompt_rc,
        max_len_q=args.max_len_q,
        max_len_d=args.max_len_d,
    )
    grad_accum_steps = max(args.grad_accum, 1)
    use_alternating_train = train_pc_ds is not None and train_rc_ds is not None
    eval_bs = args.eval_batch_size if args.eval_batch_size and args.eval_batch_size > 0 else args.batch_size
    train_loader = None
    train_pc_loader = None
    train_rc_loader = None
    if use_alternating_train:
        train_pc_loader = DataLoader(train_pc_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True, num_workers=0)
        train_rc_loader = DataLoader(train_rc_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True, num_workers=0)
    elif train_pc_ds is not None:
        train_loader = DataLoader(train_pc_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True, num_workers=0)
    else:
        train_loader = DataLoader(train_rc_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator, drop_last=False, num_workers=0)
    val_pc_loader = DataLoader(val_pc_ds if val_pc_ds is not None else [], batch_size=eval_bs, shuffle=False, collate_fn=collator, drop_last=False, num_workers=0)
    val_rc_loader = DataLoader(val_rc_ds if val_rc_ds is not None else [], batch_size=eval_bs, shuffle=False, collate_fn=collator, drop_last=False, num_workers=0)
    if use_alternating_train:
        max_loader_len = max(len(train_pc_loader), len(train_rc_loader))
        optimizer_steps_per_view = math.ceil(max_loader_len / grad_accum_steps)
        optimizer_steps_per_epoch = 2 * optimizer_steps_per_view
        total_micro_steps_per_epoch = optimizer_steps_per_epoch * grad_accum_steps
    else:
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_micro_steps_per_epoch = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = optimizer_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if use_alternating_train:
        embedder, optimizer, train_pc_loader, train_rc_loader, val_loader, val_pc_loader, val_rc_loader, scheduler = accelerator.prepare(
            embedder, optimizer, train_pc_loader, train_rc_loader, val_loader, val_pc_loader, val_rc_loader, scheduler
        )
    else:
        embedder, optimizer, train_loader, val_loader, val_pc_loader, val_rc_loader, scheduler = accelerator.prepare(
            embedder, optimizer, train_loader, val_loader, val_pc_loader, val_rc_loader, scheduler
        )

    best_val_kendall_star = float("inf")
    no_improve = 0
    global_step = 0
    optimizer_step = 0
    run_name = f"best_contrastive_{args.run_tag}" if args.run_tag else "best_contrastive"
    best_dir = os.path.join(args.output_dir, run_name)

    def evaluate_all():
        val_loss, val_acc, val_kendall, val_kendall_used, val_kendall_skipped = eval_loss_and_acc(
            embedder=embedder,
            loader=val_loader,
            accelerator=accelerator,
            temperature=args.temperature,
            inbatch_weight=args.inbatch_weight,
            pair_weight=args.pair_weight,
            pair_margin=args.pair_margin,
        )
        val_pc_loss = 0.0
        val_rc_loss = 0.0
        val_pc_acc = 0.0
        val_rc_acc = 0.0
        val_pc_kendall = float("nan")
        val_rc_kendall = float("nan")
        val_pc_kendall_used = 0
        val_rc_kendall_used = 0
        val_pc_kendall_skipped = 0
        val_rc_kendall_skipped = 0
        has_pc = val_pc_ds is not None
        has_rc = val_rc_ds is not None
        if has_pc:
            val_pc_loss, val_pc_acc, val_pc_kendall, val_pc_kendall_used, val_pc_kendall_skipped = eval_loss_and_acc(
                embedder=embedder,
                loader=val_pc_loader,
                accelerator=accelerator,
                temperature=args.temperature,
                inbatch_weight=args.inbatch_weight,
                pair_weight=args.pair_weight,
                pair_margin=args.pair_margin,
            )
        if has_rc:
            val_rc_loss, val_rc_acc, val_rc_kendall, val_rc_kendall_used, val_rc_kendall_skipped = eval_loss_and_acc(
                embedder=embedder,
                loader=val_rc_loader,
                accelerator=accelerator,
                temperature=args.temperature,
                inbatch_weight=args.inbatch_weight,
                pair_weight=args.pair_weight,
                pair_margin=args.pair_margin,
            )
        if has_pc and has_rc:
            val_loss_star = 0.5 * val_pc_loss + 0.5 * val_rc_loss
            val_acc_star = 0.5 * val_pc_acc + 0.5 * val_rc_acc
            val_kendall_star = 0.5 * val_pc_kendall + 0.5 * val_rc_kendall
        elif has_pc:
            val_loss_star = val_pc_loss
            val_acc_star = val_pc_acc
            val_kendall_star = val_pc_kendall
        else:
            val_loss_star = val_rc_loss
            val_acc_star = val_rc_acc
            val_kendall_star = val_rc_kendall
        return (
            val_loss,
            val_acc,
            val_kendall,
            val_kendall_used,
            val_kendall_skipped,
            val_pc_loss,
            val_rc_loss,
            val_loss_star,
            val_pc_acc,
            val_rc_acc,
            val_acc_star,
            val_pc_kendall,
            val_rc_kendall,
            val_kendall_star,
            val_pc_kendall_used,
            val_rc_kendall_used,
            val_pc_kendall_skipped,
            val_rc_kendall_skipped,
        )

    for epoch in range(args.epochs):
        embedder.train()
        running_loss = torch.tensor(0.0, device=accelerator.device)
        step_count = 0
        t0 = time.time()
        if use_alternating_train:
            train_pc_iter = iter(train_pc_loader)
            train_rc_iter = iter(train_rc_loader)
            pbar = tqdm(range(total_micro_steps_per_epoch), disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}/{args.epochs}")
        else:
            train_iter = iter(train_loader)
            pbar = tqdm(range(total_micro_steps_per_epoch), disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for micro_step_idx in pbar:
            if use_alternating_train:
                if micro_step_idx % 2 == 0:
                    try:
                        batch = next(train_pc_iter)
                    except StopIteration:
                        train_pc_iter = iter(train_pc_loader)
                        batch = next(train_pc_iter)
                else:
                    try:
                        batch = next(train_rc_iter)
                    except StopIteration:
                        train_rc_iter = iter(train_rc_loader)
                        batch = next(train_rc_iter)
            else:
                batch = next(train_iter)
            with accelerator.accumulate(embedder):
                q_pos = {k: v.to(accelerator.device) for k, v in batch.q_pos_input.items()}
                q_neg = {k: v.to(accelerator.device) for k, v in batch.q_neg_input.items()}
                pos = {k: v.to(accelerator.device) for k, v in batch.pos_input.items()}
                neg = {k: v.to(accelerator.device) for k, v in batch.neg_input.items()}
                q_pos_emb = embedder(q_pos)
                q_neg_emb = embedder(q_neg)
                pos_emb = embedder(pos)
                neg_emb = embedder(neg)
                pair_w = batch.pair_sample_weight.to(accelerator.device)
                loss, loss_ce, loss_pair = mixed_contrastive_loss(
                    q_pos_emb,
                    q_neg_emb,
                    pos_emb,
                    neg_emb,
                    accelerator=accelerator,
                    temperature=args.temperature,
                    inbatch_weight=args.inbatch_weight,
                    pair_weight=args.pair_weight,
                    pair_margin=args.pair_margin,
                    pair_sample_weight=pair_w,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                did_step = accelerator.sync_gradients

            step_count += 1
            global_step += 1
            if did_step:
                optimizer_step += 1
            running_loss += loss.detach()
            if accelerator.is_main_process:
                pbar.set_postfix(
                    {
                        "loss": f"{(running_loss / step_count).item():.4f}",
                        "ce": f"{loss_ce.item():.4f}",
                        "pair": f"{loss_pair.item():.4f}",
                    }
                )

            if args.eval_every_steps > 0 and did_step and optimizer_step % args.eval_every_steps == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                (
                    val_loss,
                    val_acc,
                    val_kendall,
                    val_kendall_used,
                    val_kendall_skipped,
                    val_pc_loss,
                    val_rc_loss,
                    val_loss_star,
                    val_pc_acc,
                    val_rc_acc,
                    val_acc_star,
                    val_pc_kendall,
                    val_rc_kendall,
                    val_kendall_star,
                    val_pc_kendall_used,
                    val_rc_kendall_used,
                    val_pc_kendall_skipped,
                    val_rc_kendall_skipped,
                ) = evaluate_all()
                if accelerator.is_main_process:
                    print(
                        f"[Step {optimizer_step}] val_loss={val_loss:.4f} val_pairwise_acc={val_acc:.4f} "
                        f"val_pc_loss={val_pc_loss:.4f} val_rc_loss={val_rc_loss:.4f} val_loss_star={val_loss_star:.4f} "
                        f"val_pc_acc={val_pc_acc:.4f} val_rc_acc={val_rc_acc:.4f} val_acc_star={val_acc_star:.4f} "
                        f"val_kendall={val_kendall:.4f} val_pc_kendall={val_pc_kendall:.4f} val_rc_kendall={val_rc_kendall:.4f} "
                        f"val_kendall_star={val_kendall_star:.4f} val_kendall_used={val_kendall_used} val_kendall_skipped={val_kendall_skipped} "
                        f"val_pc_kendall_used={val_pc_kendall_used} val_pc_kendall_skipped={val_pc_kendall_skipped} "
                        f"val_rc_kendall_used={val_rc_kendall_used} val_rc_kendall_skipped={val_rc_kendall_skipped}"
                    )
                val_kendall_cmp = val_kendall_star if math.isfinite(val_kendall_star) else float("inf")
                if val_kendall_cmp < best_val_kendall_star:
                    best_val_kendall_star = val_kendall_cmp
                    no_improve = 0
                    maybe_save_best(accelerator, embedder, tokenizer, args, best_dir)
                    if accelerator.is_main_process:
                        print(f"[Saved] {best_dir} (best_val_kendall_star={best_val_kendall_star:.4f})")
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        if accelerator.is_main_process:
                            print(f"[EarlyStopping] no improvement for {no_improve} evals")
                        return
                embedder.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        epoch_train_loss = reduce_mean(running_loss / max(step_count, 1), accelerator).item()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (
            val_loss,
            val_acc,
            val_kendall,
            val_kendall_used,
            val_kendall_skipped,
            val_pc_loss,
            val_rc_loss,
            val_loss_star,
            val_pc_acc,
            val_rc_acc,
            val_acc_star,
            val_pc_kendall,
            val_rc_kendall,
            val_kendall_star,
            val_pc_kendall_used,
            val_rc_kendall_used,
            val_pc_kendall_skipped,
            val_rc_kendall_skipped,
        ) = evaluate_all()
        dt = time.time() - t0
        if accelerator.is_main_process:
            print(
                f"[Epoch {epoch + 1}] train_loss={epoch_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_pairwise_acc={val_acc:.4f} "
                f"val_pc_loss={val_pc_loss:.4f} val_rc_loss={val_rc_loss:.4f} val_loss_star={val_loss_star:.4f} "
                f"val_pc_acc={val_pc_acc:.4f} val_rc_acc={val_rc_acc:.4f} val_acc_star={val_acc_star:.4f} "
                f"val_kendall={val_kendall:.4f} val_pc_kendall={val_pc_kendall:.4f} val_rc_kendall={val_rc_kendall:.4f} "
                f"val_kendall_star={val_kendall_star:.4f} val_kendall_used={val_kendall_used} val_kendall_skipped={val_kendall_skipped} "
                f"val_pc_kendall_used={val_pc_kendall_used} val_pc_kendall_skipped={val_pc_kendall_skipped} "
                f"val_rc_kendall_used={val_rc_kendall_used} val_rc_kendall_skipped={val_rc_kendall_skipped} time={dt:.1f}s"
            )
        val_kendall_cmp = val_kendall_star if math.isfinite(val_kendall_star) else float("inf")
        if val_kendall_cmp < best_val_kendall_star:
            best_val_kendall_star = val_kendall_cmp
            no_improve = 0
            maybe_save_best(accelerator, embedder, tokenizer, args, best_dir)
            if accelerator.is_main_process:
                print(f"[Saved] {best_dir} (best_val_kendall_star={best_val_kendall_star:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                if accelerator.is_main_process:
                    print(f"[EarlyStopping] no improvement for {no_improve} evals")
                return
        embedder.train()


if __name__ == "__main__":
    main()
