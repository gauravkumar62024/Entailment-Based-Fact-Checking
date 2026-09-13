#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fast semantic-chunking + retrieval (custom/langchain) with batching + claim-embedding cache.

Supports:
- LIAR-RAW
- RAWFC
- X-FACT (with multilingual sentence splitting from raw webpages)

Key speedups:
- Embed claim once
- Batched chunk embeddings
- Vectorized cosine similarity
- Optional AMP + TF32
"""

import os
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ------------------------
# Project root
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_chunking import (
    semantic_chunk_from_tokenized,
    semantic_chunk_langchain_from_tokenized,
)
from utils.load_data import load_json_or_jsonl
from utils.sentence_split import clean_web_text, split_into_sentences

MODEL_NAME_DEFAULT = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# X-FACT helpers
# ------------------------
MAX_CHARS = 200_000      # safe (≈ 200k chars)
STRIDE = 150_000         # overlap to avoid sentence cut

def webpage_to_tokenized(content: str):
    if not content or not content.strip():
        return []

    clean = clean_web_text(content)

    tokenized = []
    sent_id = 0

    for start in range(0, len(clean), STRIDE):
        chunk = clean[start:start + MAX_CHARS]
        if not chunk.strip():
            continue

        sentences = split_into_sentences(chunk)

        for sent in sentences:
            tokenized.append({
                "sentence_id": sent_id,
                "text": sent,
                "is_evidence": None
            })
            sent_id += 1

    return tokenized



def xfact_to_internal_format(xfact_dp):
    reports = []

    for src in xfact_dp.get("sources", []):
        tokenized = webpage_to_tokenized(src.get("content", ""))

        if not tokenized:
            continue

        reports.append({
            "link": src.get("source"),
            "domain": None,
            "tokenized": tokenized
        })

    return {
        "event_id": None,
        "claim": xfact_dp.get("claim", ""),
        "original_label": xfact_dp.get("label"),
        "label": xfact_dp.get("label"),
        "language": xfact_dp.get("language"),
        "explain": None,
        "reports": reports
    }

# ------------------------
# Output helpers
# ------------------------
def normalize_dataset_name(dataset: str) -> str:
    if dataset.upper() == "LIAR-RAW":
        return "LIAR-RAW"
    if dataset.lower() == "rawfc":
        return "rawfc"
    if dataset.lower() == "xfact":
        return "X-FACT"
    return dataset


def build_output_path(out_root: str, chunker: str, dataset: str, split: str) -> str:
    dataset_dir = normalize_dataset_name(dataset)
    out_dir = os.path.join(out_root, f"chunked_{chunker}", dataset_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{split}.json")

# ------------------------
# Data loading
# ------------------------
def resolve_input_path(data_root: str, dataset: str, split: str) -> str:
    ds = dataset.lower()

    if ds == "liar-raw":
        return os.path.join(data_root, "LIAR-RAW", f"{split}.json")

    if ds == "rawfc":
        return os.path.join(data_root, "rawfc", split)

    if ds == "xfact":
        base = os.path.join(data_root, "X-FACT")
        json_path = os.path.join(base, f"{split}.json")
        jsonl_path = os.path.join(base, f"{split}.jsonl")

        if os.path.exists(json_path):
            return json_path
        if os.path.exists(jsonl_path):
            return jsonl_path

        raise FileNotFoundError(
            f"X-FACT split not found: {json_path} or {jsonl_path}"
        )

    raise ValueError("dataset must be 'LIAR-RAW', 'rawfc', or 'xfact'")



def load_split_data(input_path: str, dataset: str) -> List[Dict[str, Any]]:
    if dataset.lower() == "rawfc":
        all_data = []
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith(".json"):
                all_data.extend(load_json_or_jsonl(os.path.join(input_path, fname)))
        return all_data
    return load_json_or_jsonl(input_path)

# ------------------------
# Embedding utils
# ------------------------
@torch.no_grad()
def embed_texts_batched(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 256,
    max_length: int = 256,
    use_amp: bool = False,
) -> torch.Tensor:

    if not texts:
        return torch.empty((0, 0), device=device)

    embs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        if use_amp and device.startswith("cuda"):
            with torch.cuda.amp.autocast():
                out = model(**enc)
        else:
            out = model(**enc)

        token_emb = out.last_hidden_state
        attn_mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (token_emb * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
        pooled = F.normalize(pooled, p=2, dim=1)
        embs.append(pooled)

    return torch.cat(embs, dim=0)


@torch.no_grad()
def retrieve_topk_chunks_fast(
    claim_emb: torch.Tensor,
    chunk_texts: List[str],
    chunk_dicts: List[Dict[str, Any]],
    tokenizer,
    model,
    k: int = 2,
    batch_size: int = 256,
    max_length: int = 256,
    use_amp: bool = False,
):

    if not chunk_texts:
        return []

    chunk_embs = embed_texts_batched(
        model=model,
        tokenizer=tokenizer,
        texts=chunk_texts,
        device=DEVICE,
        batch_size=batch_size,
        max_length=max_length,
        use_amp=use_amp,
    )

    scores = torch.matmul(chunk_embs, claim_emb)
    topk = min(k, scores.numel())
    idxs = torch.topk(scores, k=topk).indices.tolist()

    return [
        {
            "chunk": chunk_dicts[i],
            "chunk_text": chunk_texts[i],
            "score": float(scores[i].item()),
        }
        for i in idxs
    ]

# ------------------------
# Main processing
# ------------------------
def process_datapoint_fast(
    dp,
    tokenizer,
    model,
    k=2,
    label_key="is_evidence",
    chunker_type="custom",
    batch_size=256,
    max_length=256,
    use_amp=False,
):

    claim = dp.get("claim", "") or ""

    claim_emb = embed_texts_batched(
        model,
        tokenizer,
        [claim],
        DEVICE,
        batch_size=1,
        max_length=max_length,
        use_amp=use_amp,
    )[0]

    processed_reports = []

    for report in dp.get("reports", []):
        tokenized = report.get("tokenized", [])

        if not tokenized:
            processed_reports.append({
                "link": report.get("link"),
                "domain": report.get("domain"),
                "chunks": [],
                "retrieved_chunks": [],
            })
            continue

        if chunker_type == "custom":
            chunk_dicts, chunk_texts = semantic_chunk_from_tokenized(
                tokenized,
                tokenizer,
                model,
                label_key,
                min_chunk_sentences=2,
                alpha=0.5,
                device=DEVICE,
            )
        else:
            chunk_dicts, chunk_texts = semantic_chunk_langchain_from_tokenized(
                tokenized,
                embedding_model_name=tokenizer.name_or_path,
                min_chunk_sentences=2,
            )

        retrieved = retrieve_topk_chunks_fast(
            claim_emb,
            chunk_texts,
            chunk_dicts,
            tokenizer,
            model,
            k,
            batch_size,
            max_length,
            use_amp,
        )

        processed_reports.append({
            "link": report.get("link"),
            "domain": report.get("domain"),
            "chunks": chunk_dicts,
            "retrieved_chunks": retrieved,
        })

    return {
        "event_id": dp.get("event_id"),
        "claim": claim,
        "original_label": dp.get("original_label"),
        "label": dp.get("label"),
        "language": dp.get("language"),
        "explain": dp.get("explain"),
        "reports": processed_reports,
    }

# ------------------------
# Entry
# ------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["LIAR-RAW", "rawfc", "xfact"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--chunker", type=str, default="langchain", choices=["custom", "langchain"])
    parser.add_argument("--model_name", type=str, default=MODEL_NAME_DEFAULT)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--out_root", type=str, default="../data/processed")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--amp", action="store_true")

    args = parser.parse_args()

    if DEVICE.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(DEVICE).eval()

    data = load_split_data(
        resolve_input_path(args.data_root, args.dataset, args.split),
        args.dataset
    )

    out = []
    for dp in tqdm(data, desc=f"{args.dataset} | {args.split}"):
        if args.dataset.lower() == "xfact":
            dp = xfact_to_internal_format(dp)

        out.append(
            process_datapoint_fast(
                dp,
                tokenizer,
                model,
                k=args.k,
                chunker_type=args.chunker,
                batch_size=args.batch_size,
                max_length=args.max_length,
                use_amp=args.amp,
            )
        )

    output_path = build_output_path(
        args.out_root, args.chunker, args.dataset, args.split
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(out)} datapoints → {output_path}")


if __name__ == "__main__":
    main()
