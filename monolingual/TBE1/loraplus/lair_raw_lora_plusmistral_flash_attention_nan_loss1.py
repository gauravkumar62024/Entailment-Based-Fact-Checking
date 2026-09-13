#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA+ approach for coarse-to-fine retrieval on LAIR-RAW.
- Still uses PEFT for LoRA injection (LoraConfig, get_peft_model).
- Uses lora_plus to create a custom optimizer with special LR for LoRA B-matrices.
- Keeps your existing custom training loop (coarse-fine retrieval).
"""

import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

# --- Hugging Face Transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_scheduler,
    set_seed,
)

# --- PEFT for LoRA injection
from peft import LoraConfig, get_peft_model

# --- LoRA+ for the specialized optimizer
#     (This is what implements "LoRA+": different LR for lora_B vs base params)
from lora_plus import create_loraplus_optimizer


# ==============================
# 1. Data Classes and Loaders
# ==============================

@dataclass
class ReportSentence:
    sent_id: str
    text: str
    is_evidence: int  # 0 or 1

@dataclass
class Report:
    report_id: int
    content: str
    sentences: List[ReportSentence]

@dataclass
class ClaimExample:
    event_id: str
    claim_text: str
    label: str
    reports: List[Report]

def load_lair_raw(json_path: str) -> List[ClaimExample]:
    """Load LAIR-RAW data from JSON, returning a list of ClaimExample."""
    if not os.path.isfile(json_path):
        print(f"[WARN] File {json_path} does not exist. Returning empty list.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data:
        event_id   = item.get("event_id", "")
        claim_text = item.get("claim", "")
        label      = item.get("label", "unknown")
        r_list     = []
        for rep in item.get("reports", []):
            r_id    = rep.get("report_id", -1)
            content = rep.get("content", "")
            sents   = []
            tokenized = rep.get("tokenized", [])
            for i, tok_obj in enumerate(tokenized):
                sid = f"{event_id}__{r_id}__sent{i}"
                sents.append(
                    ReportSentence(
                        sent_id     = sid,
                        text        = tok_obj.get("sent", ""),
                        is_evidence = tok_obj.get("is_evidence", 0)
                    )
                )
            r_list.append(Report(report_id=r_id, content=content, sentences=sents))
        examples.append(
            ClaimExample(
                event_id   = event_id,
                claim_text = claim_text,
                label      = label,
                reports    = r_list
            )
        )
    return examples[:20]

# ==============================
# 2. Model + PEFT-LoRA Injection
# ==============================

MODEL_NAME = "neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model_lora_plus(
    model_name: str = MODEL_NAME,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules = ("q_proj", "v_proj"),
    use_fp16: bool = True
):
    """
    1) Load base model & tokenizer
    2) Add pad token if missing
    3) Create standard PEFT LoraConfig
    4) Use PEFT's get_peft_model() to inject LoRA
    5) Return the LoRA-adapted model
    """
    print(f"Loading base model `{model_name}` with LoRA+ approach ...")

    load_kwargs = {"trust_remote_code": True}
    if use_fp16:
        load_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False, force_download=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_config = AutoConfig.from_pretrained(model_name, **load_kwargs)
    base_model = AutoModel.from_pretrained(model_name, config=base_config, **load_kwargs)

    # If we added a pad token, need to resize embedding
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    # Step 3) Standard PEFT LoraConfig (this is still "LoRA").
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        # 'use_original_init' or 'modules_to_save' can be set if needed
    )

    # Step 4) get_peft_model to inject LoRA
    lora_model = get_peft_model(base_model, config)
    lora_model = lora_model.to(device)

    return lora_model, tokenizer, base_config

# ==============================
# 3. Embedding Function
# ==============================

def embed_with_lora_plus(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    prefix: str = "query: ",
    batch_size: int = 128,
    max_length: int = 256
) -> np.ndarray:
    """
    Encode texts using the LoRA-adapted Mistral model (PEFT).
    Returns a numpy array: (len(texts), hidden_dim).
    """
    model.eval()
    all_embs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        # Replace empty text with placeholder
        batch_texts = [t if t.strip() else "N/A" for t in batch_texts]
        batch_prefixed = [prefix + t for t in batch_texts]

        enc = tokenizer(
            batch_prefixed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            last_hidden_state = outputs.last_hidden_state  # [B, L, hidden_dim]
            # For a LLaMA/Mistral-based model, we can use the first token's embedding
            embs = last_hidden_state[:, 0, :].float().cpu().numpy()
        all_embs.append(embs)

    return np.concatenate(all_embs, axis=0)

# ==============================
# 4. Coarse-Fine Architecture
# ==============================

class CoarseFineRetriever(nn.Module):
    """
    Simple 2-head architecture: doc-level + sentence-level classification.
    `hidden_dim` must match your embedding dimension from the base LLM.
    """
    def __init__(self, hidden_dim: int):
        super(CoarseFineRetriever, self).__init__()
        self.doc_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.sent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_doc(self, doc_embs: torch.Tensor):
        return self.doc_classifier(doc_embs)

    def forward_sent(self, sent_embs: torch.Tensor):
        return self.sent_classifier(sent_embs)

# ==============================
# 5. Training Loop (LoRA+)
# ==============================

@dataclass
class Stats:
    total_loss: float = 0.0

def doc_label_for_report(rep: Report) -> int:
    """doc label=1 if any sentence has is_evidence=1, else 0"""
    return 1 if any(s.is_evidence == 1 for s in rep.sentences) else 0

def get_oracle_sentence_ids(rep: Report) -> List[int]:
    """Return list of 0/1 for each sentence in a single report."""
    return [s.is_evidence for s in rep.sentences]

TOP_K_DOC = 20

def train_coarse_fine(
    model: CoarseFineRetriever,
    base_llm: nn.Module,
    tokenizer,
    train_data: List[ClaimExample],
    num_epochs: int = 3,
    batch_size: int = 640,
    lr: float = 1e-5,
    alpha_doc: float = 1.0,
    alpha_sent: float = 1.0,
    output_dir: str = "outputs_coarse_fine"
):
    """
    Simplified training loop that freezes base model weights and trains
    only LoRA adapter and classifier parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_seed(42)

    # Freeze base model weights (base_llm)
    for param in base_llm.parameters():
        param.requires_grad = False

    # Ensure only LoRA adapter and classifier weights are trainable
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # Use AdamW optimizer only for trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0)

    # Scheduler for learning rate
    total_steps = len(train_data) * num_epochs
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    # Loss function
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # Start training
    model.train()
    base_llm.train()

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1} / {num_epochs} ===")
        np.random.shuffle(train_data)

        total_loss = 0.0
        for step, claim_ex in enumerate(train_data, 1):
            if not claim_ex.claim_text.strip():
                continue

            optimizer.zero_grad()

            # 1) Embed claim
            claim_emb = embed_with_lora_plus(
                base_llm, tokenizer, [claim_ex.claim_text], prefix="query: ", batch_size=1
            )
            claim_emb_t = torch.from_numpy(claim_emb).float().to(device)

            # 2) Embed documents
            doc_texts = [r.content if r.content.strip() else "N/A" for r in claim_ex.reports]
            if not doc_texts:
                continue

            doc_embs = embed_with_lora_plus(
                base_llm, tokenizer, doc_texts, prefix="passage: ", batch_size=30
            )
            doc_embs_t = torch.from_numpy(doc_embs).float().to(device)

            doc_labels = [doc_label_for_report(r) for r in claim_ex.reports]
            doc_labels_t = torch.tensor(doc_labels, dtype=torch.float, device=device).unsqueeze(-1)

            # 3) Forward pass (doc-level)
            doc_logits = model.forward_doc(doc_embs_t)
            doc_loss = bce_loss_fn(doc_logits, doc_labels_t)

            # 4) Pick top-K docs
            with torch.no_grad():
                doc_probs = torch.sigmoid(doc_logits).squeeze(-1)
            K = min(TOP_K_DOC, len(doc_texts))
            sorted_indices = torch.argsort(doc_probs, descending=True)
            topk_indices = sorted_indices[:K]

            # 5) Gather sentences from top-K docs
            all_sent_texts = []
            all_sent_labels = []
            for idx in topk_indices:
                doc_obj = claim_ex.reports[idx.item()]
                all_sent_labels.extend(get_oracle_sentence_ids(doc_obj))
                for s_obj in doc_obj.sentences:
                    all_sent_texts.append(s_obj.text.strip() if s_obj.text.strip() else "N/A")

            # If no sentences, only backpropagate doc_loss
            if not all_sent_texts:
                doc_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += doc_loss.item()
                continue

            # 6) Embed sentences
            sent_embs = embed_with_lora_plus(
                base_llm, tokenizer, all_sent_texts, prefix="passage: ", batch_size=20
            )
            sent_embs_t = torch.from_numpy(sent_embs).float().to(device)
            sent_labels_t = torch.tensor(all_sent_labels, dtype=torch.float, device=device).unsqueeze(-1)

            # 7) Forward pass (sentence-level)
            sent_logits = model.forward_sent(sent_embs_t)
            sent_loss = bce_loss_fn(sent_logits, sent_labels_t)

            # 8) Combine and backpropagate
            loss = alpha_doc * doc_loss + alpha_sent * sent_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} finished. avg loss = {total_loss / len(train_data):.4f}")

    # Save final model
    final_path = os.path.join(output_dir, "final_coarse_fine_model_lora_plus_mistral.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")


# ==============================
# 6. Evaluation
# ==============================

def precision_recall_f1_at_k(ranked_ids: List[str], gold_ids: set, k: int):
    if k > len(ranked_ids):
        k = len(ranked_ids)
    retrieved = set(ranked_ids[:k])
    if len(gold_ids) == 0:
        return 0.0, 0.0, 0.0
    tp = len(retrieved.intersection(gold_ids))
    prec = tp / float(k) if k > 0 else 0.0
    rec  = tp / float(len(gold_ids))
    if prec + rec == 0:
        return 0.0, 0.0, 0.0
    f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1

def compute_mrr(ranked_ids: List[str], gold_ids: set):
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold_ids:
            return 1.0/float(i)
    return 0.0

def compute_ndcg(ranked_ids: List[str], gold_ids: set):
    dcg = 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i+1)
    ideal_count = min(len(gold_ids), len(ranked_ids))
    idcg = 0.0
    for i in range(1, ideal_count+1):
        idcg += 1.0 / math.log2(i+1)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_coarse_fine(
    model: CoarseFineRetriever,
    base_llm: nn.Module,
    tokenizer,
    eval_data: List[ClaimExample],
    output_json="eval_results_lora_plus.json"
):
    model.eval()
    base_llm.eval()

    K_VALUES = [5,10,15,20]
    doc_storage = {f"{m}@{k}":[] for k in K_VALUES for m in ["P","R","F1","nDCG","MRR"]}
    sent_storage= {f"{m}@{k}":[] for k in K_VALUES for m in ["P","R","F1","nDCG","MRR"]}

    results = []
    for claim_ex in tqdm(eval_data, desc="Evaluating"):
        claim_text = claim_ex.claim_text
        gold_docs  = set()
        gold_sents = set()

        for r in claim_ex.reports:
            if any(s.is_evidence==1 for s in r.sentences):
                gold_docs.add(r.report_id)
            for s in r.sentences:
                if s.is_evidence==1:
                    gold_sents.add(s.sent_id)

        if not claim_text.strip():
            continue

        # embed claim
        claim_emb = embed_with_lora_plus(base_llm, tokenizer,
                                         [claim_text], prefix="query: ")
        c_emb_t = torch.from_numpy(claim_emb).float().to(device)

        doc_texts = [r.content if r.content.strip() else "N/A"
                     for r in claim_ex.reports]
        doc_ids   = [r.report_id for r in claim_ex.reports]
        if not doc_texts:
            continue

        doc_embs = embed_with_lora_plus(base_llm, tokenizer,
                                        doc_texts, prefix="passage: ")
        doc_embs_t = torch.from_numpy(doc_embs).float().to(device)

        with torch.no_grad():
            logits_doc = model.forward_doc(doc_embs_t).squeeze(-1)
            scores_doc = torch.sigmoid(logits_doc).cpu().numpy()

        # Sort docs by predicted relevance
        doc_score_pairs = sorted(zip(doc_ids, scores_doc),
                                 key=lambda x: x[1], reverse=True)
        ranked_doc_ids = [d[0] for d in doc_score_pairs]

        # Evaluate doc-level
        doc_metrics = {}
        for k_ in K_VALUES:
            prec, rec, f1_ = precision_recall_f1_at_k(ranked_doc_ids,
                                                      gold_docs, k_)
            ndcg_ = compute_ndcg(ranked_doc_ids[:k_], gold_docs)
            mrr_  = compute_mrr(ranked_doc_ids[:k_], gold_docs)

            doc_metrics[f"P@{k_}"]   = prec
            doc_metrics[f"R@{k_}"]   = rec
            doc_metrics[f"F1@{k_}"]  = f1_
            doc_metrics[f"nDCG@{k_}"]= ndcg_
            doc_metrics[f"MRR@{k_}"] = mrr_

            doc_storage[f"P@{k_}"].append(prec)
            doc_storage[f"R@{k_}"].append(rec)
            doc_storage[f"F1@{k_}"].append(f1_)
            doc_storage[f"nDCG@{k_}"].append(ndcg_)
            doc_storage[f"MRR@{k_}"].append(mrr_)

        # top-K doc => sentences
        topK = min(TOP_K_DOC, len(doc_score_pairs))
        topk_docs = doc_score_pairs[:topK]
        topk_doc_ids = [x[0] for x in topk_docs]

        candidate_sents=[]
        for r in claim_ex.reports:
            if r.report_id in topk_doc_ids:
                candidate_sents.extend(r.sentences)

        if not candidate_sents:
            # no sents
            sent_metrics={}
            for k_ in K_VALUES:
                sent_metrics[f"P@{k_}"]=0.0
                sent_metrics[f"R@{k_}"]=0.0
                sent_metrics[f"F1@{k_}"]=0.0
                sent_metrics[f"nDCG@{k_}"]=0.0
                sent_metrics[f"MRR@{k_}"]=0.0

            ranked_docs_json = []
            for rank_i,(did,score) in enumerate(doc_score_pairs,1):
                ranked_docs_json.append({
                    "report_id": did,
                    "score": score,
                    "rank": rank_i,
                    "is_relevant": (did in gold_docs),
                })

            results.append({
                "event_id": claim_ex.event_id,
                "doc_metrics": doc_metrics,
                "sent_metrics": sent_metrics,
                "ranked_docs": ranked_docs_json,
                "ranked_sents": [],
            })
            continue

        sent_ids = [s.sent_id for s in candidate_sents]
        sent_texts = [s.text.strip() if s.text.strip() else "N/A"
                      for s in candidate_sents]

        sent_embs = embed_with_lora_plus(base_llm, tokenizer,
                                         sent_texts, prefix="passage: ")
        s_embs_t  = torch.from_numpy(sent_embs).float().to(device)

        with torch.no_grad():
            logits_sents = model.forward_sent(s_embs_t).squeeze(-1)
            scores_sents = torch.sigmoid(logits_sents).cpu().numpy()

        sent_score_pairs = sorted(zip(sent_ids, scores_sents),
                                  key=lambda x:x[1], reverse=True)
        ranked_sent_ids = [x[0] for x in sent_score_pairs]

        # Evaluate sentence-level
        sent_metrics={}
        for k_ in K_VALUES:
            prec, rec, f1_= precision_recall_f1_at_k(ranked_sent_ids,
                                                     gold_sents, k_)
            ndcg_= compute_ndcg(ranked_sent_ids[:k_], gold_sents)
            mrr_ = compute_mrr(ranked_sent_ids[:k_], gold_sents)

            sent_metrics[f"P@{k_}"]   = prec
            sent_metrics[f"R@{k_}"]   = rec
            sent_metrics[f"F1@{k_}"]  = f1_
            sent_metrics[f"nDCG@{k_}"] = ndcg_
            sent_metrics[f"MRR@{k_}"]  = mrr_

            sent_storage[f"P@{k_}"].append(prec)
            sent_storage[f"R@{k_}"].append(rec)
            sent_storage[f"F1@{k_}"].append(f1_)
            sent_storage[f"nDCG@{k_}"].append(ndcg_)
            sent_storage[f"MRR@{k_}"].append(mrr_)

        # Save retrieval ranking
        ranked_docs_json=[]
        for rank_i,(did,score) in enumerate(doc_score_pairs,1):
            ranked_docs_json.append({
                "report_id": did,
                "score": score,
                "rank": rank_i,
                "is_relevant": (did in gold_docs),
            })
        ranked_sents_json=[]
        for rank_i,(sid,score) in enumerate(sent_score_pairs,1):
            ranked_sents_json.append({
                "sentence_id": sid,
                "score": score,
                "rank": rank_i,
                "is_relevant": (sid in gold_sents),
            })

        results.append({
            "event_id": claim_ex.event_id,
            "doc_metrics": doc_metrics,
            "sent_metrics": sent_metrics,
            "ranked_docs": ranked_docs_json,
            "ranked_sents": ranked_sents_json,
        })

    # Summaries
    doc_avg = {}
    sent_avg= {}
    for k_ in K_VALUES:
        for m in ["P","R","F1","nDCG","MRR"]:
            arr_d = doc_storage[f"{m}@{k_}"]
            arr_s = sent_storage[f"{m}@{k_}"]
            doc_avg[f"{m}@{k_}"]  = float(np.mean(arr_d)) if arr_d else 0.0
            sent_avg[f"{m}@{k_}"] = float(np.mean(arr_s)) if arr_s else 0.0

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() 
                  if isinstance(o, (np.float32, np.int32)) else o)

    print(f"\nSaved eval results to {output_json}")

    print("=== DOC-LEVEL ===")
    for k_ in K_VALUES:
        print(f"K={k_} => "
              f"P={doc_avg[f'P@{k_}']:.4f}, "
              f"R={doc_avg[f'R@{k_}']:.4f}, "
              f"F1={doc_avg[f'F1@{k_}']:.4f}, "
              f"nDCG={doc_avg[f'nDCG@{k_}']:.4f}, "
              f"MRR={doc_avg[f'MRR@{k_}']:.4f}")

    print("=== SENT-LEVEL ===")
    for k_ in K_VALUES:
        print(f"K={k_} => "
              f"P={sent_avg[f'P@{k_}']:.4f}, "
              f"R={sent_avg[f'R@{k_}']:.4f}, "
              f"F1={sent_avg[f'F1@{k_}']:.4f}, "
              f"nDCG={sent_avg[f'nDCG@{k_}']:.4f}, "
              f"MRR={sent_avg[f'MRR@{k_}']:.4f}")

    return {"doc_avg": doc_avg, "sent_avg": sent_avg}

# ==============================
# 7. Main
# ==============================

def main():
    train_file = "./../val.json"
    val_file   = "./../val.json"
    test_file  = "./../test.json"
    output_dir = "outputs_coarse_fine"
    model_path = os.path.join(output_dir, "final_coarse_fine_model_lora_plus_mistral.pt")

    print("Loading LAIR-RAW data ...")
    train_data = load_lair_raw(train_file)
    val_data   = load_lair_raw(val_file)
    test_data  = load_lair_raw(test_file)

    # 1) Setup LoRA+ (actually standard LoRA from PEFT + custom optimizer approach)
    base_llm, tokenizer, base_config = setup_model_lora_plus(
        model_name  = MODEL_NAME,
        r           = 8,
        lora_alpha  = 16,
        lora_dropout= 0.05,
        target_modules=["q_proj","v_proj"],
        use_fp16=True
    )
    hidden_dim = base_config.hidden_size
    print(f"Detected hidden_size = {hidden_dim}")

    # 2) Our two-headed classifier
    coarse_fine_model = CoarseFineRetriever(hidden_dim=hidden_dim).to(device)

    # 3) Train if not already done
    if os.path.exists(model_path):
        print(f"Found checkpoint at {model_path}. Loading & skipping training.")
        coarse_fine_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Starting coarse-to-fine training with LoRA+ optimizer ...")
        train_coarse_fine(
    model=coarse_fine_model,
    base_llm=base_llm,
    tokenizer=tokenizer,
    train_data=train_data,
    num_epochs=1,
    batch_size=640,
    lr=1e-5,
    alpha_doc=1.0,
    alpha_sent=1.0,
    output_dir=output_dir
)

    # 4) Evaluate
    print("\nEvaluating on val set ...")
    val_metrics = evaluate_coarse_fine(
        model       = coarse_fine_model,
        base_llm    = base_llm,
        tokenizer   = tokenizer,
        eval_data   = val_data,
        output_json = "val_results_lora_plus.json"
    )
    print("Val metrics =>", val_metrics)

    print("\nEvaluating on test set ...")
    test_metrics = evaluate_coarse_fine(
        model       = coarse_fine_model,
        base_llm    = base_llm,
        tokenizer   = tokenizer,
        eval_data   = test_data,
        output_json = "test_results_lora_plus.json"
    )
    print("Test metrics =>", test_metrics)


if __name__ == "__main__":
    main()

