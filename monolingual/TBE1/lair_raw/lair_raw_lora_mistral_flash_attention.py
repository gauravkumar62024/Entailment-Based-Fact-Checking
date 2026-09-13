#!/usr/bin/env python
# -*- coding: utf-8 -*-
 #"neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
    #"neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    #"neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
    #"neuralmagic/gemma-2-9b-it-quantized.w4a16",
    #"neuralmagic/gemma-2-9b-it-quantized.w8a8",
    #"neuralmagic/gemma-2-9b-it-FP8",
    #"neuralmagic/Qwen2-7B-Instruct-quantized.w4a16",
    #"neuralmagic/Qwen2-7B-Instruct-quantized.w8a8",
    #"neuralmagic/Qwen2-7B-Instruct-FP8",
    #"neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w4a16",
    #"neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a8",
    #"neuralmagic/Mistral-7B-Instruct-v0.3-FP8",
    #"/data2/Gaurav/agent/vllm_experiments/models/falcon-7b-instruct-INT8",
    #"/data2/Gaurav/agent/vllm_experiments/models/falcon-7b-instruct-W4A16-G128",
        # Add more models as needed
"""
Final: Fine-tuning a Mistral-based model with LoRA for a coarse-to-fine retrieval
on LAIR-RAW. Fixes the shape mismatch by dynamically setting hidden_dim = base_model.config.hidden_size.
Also includes pad token logic, zero-doc checks, etc.
"""

import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List
from dataclasses import dataclass

# Hugging Face Transformers & PEFT for LoRA
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_scheduler,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

import torch.nn as nn
import torch.nn.functional as F

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
    return examples

# ==============================
# 2. Model + LoRA Setup
# ==============================

MODEL_NAME = "neuralmagic/Mistral-7B-Instruct-v0.3-quantized.w8a8"  # or "your-organization/mistral-instruct-v3-7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model_lora(
    model_name:str = MODEL_NAME,
    r:int = 8,
    alpha:int = 16,
    dropout:float = 0.05,
    task_type = TaskType.FEATURE_EXTRACTION
):
    """
    1) Load model & tokenizer
    2) Add pad token if missing
    3) Create LoRA config
    4) Wrap model with LoRA
    """
    print(f"Loading base model `{model_name}` ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 1. Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    base_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # 2. Load base model. For large LLM you might do device_map="auto", etc.
    base_model = AutoModel.from_pretrained(
        model_name,
        config=base_config,
        trust_remote_code=True
    )

    # If we added a new pad token, let's resize embedding
    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))

    # 3. LoRA config
    lora_config = LoraConfig(
        r = r,
        lora_alpha = alpha,
        target_modules = ["q_proj","v_proj"],  # adapt if needed
        lora_dropout = dropout,
        bias = "none",
        task_type = task_type
    )
    # 4. Wrap with LoRA
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)

    return model, tokenizer, base_config

# ==============================
# 3. Embedding Function
# ==============================

def embed_with_lora(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    prefix:str = "query: ",
    batch_size:int=512,
    max_length:int=256
) -> np.ndarray:
    """
    Encode texts using the LoRA-adapted Mistral/E5 model.
    Returns a numpy array: (len(texts), hidden_dim).
    """
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Replace any empty text with a placeholder
        batch_texts = [t if t.strip() else "N/A" for t in batch_texts]
        batch_prefixed = [prefix + t for t in batch_texts]

        enc = tokenizer(
            batch_prefixed,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)  # last_hidden_state: [B, L, hidden_dim]
            last_hidden_state = outputs.last_hidden_state
            # E5 recommends using the first token's embedding. Mistral is LLaMA-based, so this is a valid approach:
            embs = last_hidden_state[:, 0, :].cpu().numpy()
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
    def __init__(self, hidden_dim:int):
        super(CoarseFineRetriever, self).__init__()
        self.doc_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)  # binary
        )
        self.sent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, 1)  # binary
        )

    def forward_doc(self, claim_embs:torch.Tensor, doc_embs:torch.Tensor):
        """
        claim_embs: [num_docs, hidden_dim]
        doc_embs:   [num_docs, hidden_dim]
        We can do any combination we want. Let's do doc_embs alone.
        """
        logits = self.doc_classifier(doc_embs)
        return logits

    def forward_sent(self, claim_embs:torch.Tensor, sent_embs:torch.Tensor):
        """Similarly, do sent_embs alone for demonstration."""
        logits = self.sent_classifier(sent_embs)
        return logits

# ==============================
# 5. Training Loop
# ==============================

def doc_label_for_report(rep: Report) -> int:
    """doc label=1 if any sentence in it has is_evidence=1, else 0"""
    for s in rep.sentences:
        if s.is_evidence == 1:
            return 1
    return 0

def get_oracle_sentence_ids(rep: Report) -> List[int]:
    """Return list of 0/1 for each sentence in a single report."""
    return [s.is_evidence for s in rep.sentences]

TOP_K_DOC = 20

def train_coarse_fine(
    model: CoarseFineRetriever, 
    base_llm: nn.Module,
    tokenizer,
    train_data: List[ClaimExample],
    num_epochs=3,
    batch_size=640,
    lr=1e-5,
    alpha_doc=1.0,
    alpha_sent=1.0,
    output_dir="outputs_coarse_fine"
):
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_data)*num_epochs
    )
    bce_loss_fn = nn.BCEWithLogitsLoss()

    set_seed(42)
    model.train()

    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1} / {num_epochs} ===")
        np.random.shuffle(train_data)

        total_loss = 0.0
        for step, claim_ex in enumerate(train_data, 1):
            # skip empty claim
            if not claim_ex.claim_text.strip():
                continue

            optimizer.zero_grad()

            # 1) embed claim
            claim_emb = embed_with_lora(base_llm, tokenizer, [claim_ex.claim_text], prefix="query: ", batch_size=1)
            claim_emb_tensor = torch.from_numpy(claim_emb).float().to(device)
            # shape [1, hidden_dim]

            # 2) gather docs
            doc_texts = [r.content if r.content.strip() else "N/A" for r in claim_ex.reports]
            if len(doc_texts) == 0:
                continue

            # embed docs
            doc_embs = embed_with_lora(base_llm, tokenizer, doc_texts, prefix="passage: ", batch_size=30)
            doc_embs_tensor = torch.from_numpy(doc_embs).float().to(device)
            # shape [num_docs, hidden_dim]

            # doc labels
            doc_labels = [doc_label_for_report(r) for r in claim_ex.reports]
            doc_labels_tensor = torch.tensor(doc_labels, dtype=torch.float, device=device).unsqueeze(-1)
            # shape [num_docs, 1]

            # forward doc-level
            # We'll replicate claim embedding for each doc if we want to combine,
            # but here we do doc_embs alone.
            doc_logits = model.forward_doc(None, doc_embs_tensor)
            # shape [num_docs, 1]

            doc_loss = bce_loss_fn(doc_logits, doc_labels_tensor)

            # pick top-K by predicted doc relevance
            with torch.no_grad():
                doc_probs = torch.sigmoid(doc_logits).squeeze(-1)  # [num_docs]
            K = min(TOP_K_DOC, len(doc_texts))
            sorted_indices = torch.argsort(doc_probs, descending=True)
            topk_indices   = sorted_indices[:K]

            # 3) sentence-level classification
            all_sent_texts  = []
            all_sent_labels = []
            for idx in topk_indices:
                doc_obj = claim_ex.reports[idx.item()]
                s_labels = get_oracle_sentence_ids(doc_obj)
                for s_obj in doc_obj.sentences:
                    txt = s_obj.text if s_obj.text.strip() else "N/A"
                    all_sent_texts.append(txt)
                all_sent_labels.extend(s_labels)

            if len(all_sent_texts) == 0:
                # no sentences => only doc loss
                doc_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += doc_loss.item()
                if step % 10 == 0:
                    print(f"[{step}/{len(train_data)}] doc_loss={doc_loss.item():.4f}, sent_loss=0.0, total={doc_loss.item():.4f}")
                continue

            # embed sentences
            sent_embs = embed_with_lora(base_llm, tokenizer, all_sent_texts, prefix="passage: ", batch_size=20)
            sent_embs_tensor = torch.from_numpy(sent_embs).float().to(device)
            # shape [num_sents, hidden_dim]

            sent_labels_tensor = torch.tensor(all_sent_labels, dtype=torch.float, device=device).unsqueeze(-1)
            # shape [num_sents, 1]

            # forward
            sent_logits = model.forward_sent(None, sent_embs_tensor)
            sent_loss = bce_loss_fn(sent_logits, sent_labels_tensor)

            # combine
            loss = alpha_doc*doc_loss + alpha_sent*sent_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 10 == 0:
                print(f"[{step}/{len(train_data)}] doc_loss={doc_loss.item():.4f}, sent_loss={sent_loss.item():.4f}, total={loss.item():.4f}")

        epoch_loss = total_loss / max(1, len(train_data))
        print(f"Epoch {epoch+1} finished. avg loss = {epoch_loss:.4f}")

    final_path = os.path.join(output_dir, "final_coarse_fine_model_lora_mistral.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Model saved to {final_path}")

# ==============================
# 6. Evaluation
# ==============================

def precision_recall_f1_at_k(ranked_ids:List[str], gold_ids:set, k:int):
    if k > len(ranked_ids):
        k = len(ranked_ids)
    retrieved = set(ranked_ids[:k])
    if len(gold_ids) == 0:
        return None, None, None
    tp = len(retrieved.intersection(gold_ids))
    prec = tp / float(k) if k>0 else 0.0
    rec  = tp / float(len(gold_ids)) if len(gold_ids)>0 else 0.0
    if prec+rec==0:
        f1=0.0
    else:
        f1 = 2*prec*rec/(prec+rec)
    return prec, rec, f1

def compute_mrr(ranked_ids:List[str], gold_ids:set):
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold_ids:
            return 1.0/float(i)
    return 0.0

def compute_ndcg(ranked_ids:List[str], gold_ids:set):
    dcg = 0.0
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in gold_ids:
            dcg += 1.0 / math.log2(i+1)
    ideal_count = min(len(gold_ids), len(ranked_ids))
    idcg = 0.0
    for i in range(1, ideal_count+1):
        idcg += 1.0 / math.log2(i+1)
    return dcg/idcg if idcg>0 else 0.0

def evaluate_coarse_fine(
    model: CoarseFineRetriever,
    base_llm: nn.Module,
    tokenizer,
    eval_data: List[ClaimExample],
    output_json="eval_results_coarse_fine.json"
):
    model.eval()

    K_VALUES = [5,10,15,20]
    doc_storage = {f"{m}@{k}":[] for k in K_VALUES for m in ["P","R","F1","nDCG","MRR"]}
    sent_storage= {f"{m}@{k}":[] for k in K_VALUES for m in ["P","R","F1","nDCG","MRR"]}

    results = []
    for claim_ex in tqdm(eval_data, desc="Evaluating"):
        claim_text = claim_ex.claim_text
        # gold doc IDs
        gold_docs = set()
        for r in claim_ex.reports:
            if any(s.is_evidence==1 for s in r.sentences):
                gold_docs.add(r.report_id)
        # gold sent IDs
        gold_sents = set()
        for r in claim_ex.reports:
            for s in r.sentences:
                if s.is_evidence==1:
                    gold_sents.add(s.sent_id)

        if not claim_text.strip():
            # empty claim => skip
            continue

        # embed claim
        claim_emb = embed_with_lora(base_llm, tokenizer, [claim_text], prefix="query: ")
        c_emb_t = torch.from_numpy(claim_emb).float().to(device)

        doc_texts = [r.content if r.content.strip() else "N/A" for r in claim_ex.reports]
        doc_ids   = [r.report_id for r in claim_ex.reports]
        if len(doc_texts)==0:
            # skip
            continue

        doc_embs = embed_with_lora(base_llm, tokenizer, doc_texts, prefix="passage: ")
        doc_embs_t = torch.from_numpy(doc_embs).float().to(device)

        with torch.no_grad():
            # doc-level forward
            logits_doc = model.forward_doc(None, doc_embs_t).squeeze(-1)
            scores_doc = torch.sigmoid(logits_doc).cpu().numpy()

        doc_score_pairs = sorted(zip(doc_ids, scores_doc), key=lambda x:x[1], reverse=True)
        ranked_doc_ids = [d[0] for d in doc_score_pairs]

        # Evaluate doc-level
        doc_metrics = {}
        for k_ in K_VALUES:
            prec, rec, f1_ = precision_recall_f1_at_k(ranked_doc_ids, gold_docs, k_)
            ndcg_ = compute_ndcg(ranked_doc_ids[:k_], gold_docs)
            mrr_  = compute_mrr(ranked_doc_ids[:k_], gold_docs)
            doc_metrics[f"P@{k_}"]   = prec
            doc_metrics[f"R@{k_}"]   = rec
            doc_metrics[f"F1@{k_}"]  = f1_
            doc_metrics[f"nDCG@{k_}"]= ndcg_
            doc_metrics[f"MRR@{k_}"] = mrr_
            if prec is not None:
                doc_storage[f"P@{k_}"].append(prec)
            if rec is not None:
                doc_storage[f"R@{k_}"].append(rec)
            if f1_ is not None:
                doc_storage[f"F1@{k_}"].append(f1_)
            doc_storage[f"nDCG@{k_}"].append(ndcg_)
            doc_storage[f"MRR@{k_}"].append(mrr_)

        # top-K doc retrieval for sentence
        topK = min(TOP_K_DOC, len(doc_score_pairs))
        topk_docs = doc_score_pairs[:topK]
        topk_doc_ids = [x[0] for x in topk_docs]

        candidate_sents=[]
        for r in claim_ex.reports:
            if r.report_id in topk_doc_ids:
                candidate_sents.extend(r.sentences)

        if len(candidate_sents)==0:
            # no sents
            sent_metrics={}
            for k_ in K_VALUES:
                sent_metrics[f"P@{k_}"]=None
                sent_metrics[f"R@{k_}"]=None
                sent_metrics[f"F1@{k_}"]=None
                sent_metrics[f"nDCG@{k_}"]=0.0
                sent_metrics[f"MRR@{k_}"]=0.0
            # store
            ranked_docs_json = []
            for rank_i,(did,score) in enumerate(doc_score_pairs,1):
                ranked_docs_json.append({
                    "report_id": did,
                    "score": score,
                    "rank": rank_i,
                    "is_relevant": (did in gold_docs)
                })
            results.append({
                "event_id": claim_ex.event_id,
                "doc_metrics": doc_metrics,
                "sent_metrics": sent_metrics,
                "ranked_docs": ranked_docs_json,
                "ranked_sents": []
            })
            continue

        # embed sentences
        sent_ids = [s.sent_id for s in candidate_sents]
        sent_texts = [s.text if s.text.strip() else "N/A" for s in candidate_sents]
        sent_embs = embed_with_lora(base_llm, tokenizer, sent_texts, prefix="passage: ")
        s_embs_t = torch.from_numpy(sent_embs).float().to(device)

        with torch.no_grad():
            logits_sents = model.forward_sent(None, s_embs_t).squeeze(-1)
            scores_sents = torch.sigmoid(logits_sents).cpu().numpy()

        sent_score_pairs = sorted(zip(sent_ids,scores_sents), key=lambda x:x[1], reverse=True)
        ranked_sent_ids = [x[0] for x in sent_score_pairs]

        # Evaluate sentence-level
        sent_metrics={}
        for k_ in K_VALUES:
            prec, rec, f1_= precision_recall_f1_at_k(ranked_sent_ids, gold_sents, k_)
            ndcg_= compute_ndcg(ranked_sent_ids[:k_], gold_sents)
            mrr_ = compute_mrr(ranked_sent_ids[:k_], gold_sents)
            sent_metrics[f"P@{k_}"]   = prec
            sent_metrics[f"R@{k_}"]   = rec
            sent_metrics[f"F1@{k_}"]  = f1_
            sent_metrics[f"nDCG@{k_}"]= ndcg_
            sent_metrics[f"MRR@{k_}"] = mrr_
            if prec is not None:
                sent_storage[f"P@{k_}"].append(prec)
            if rec is not None:
                sent_storage[f"R@{k_}"].append(rec)
            if f1_ is not None:
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
                "is_relevant": (did in gold_docs)
            })
        ranked_sents_json=[]
        for rank_i,(sid,score) in enumerate(sent_score_pairs,1):
            ranked_sents_json.append({
                "sentence_id": sid,
                "score": score,
                "rank": rank_i,
                "is_relevant": (sid in gold_sents)
            })

        results.append({
            "event_id": claim_ex.event_id,
            "doc_metrics": doc_metrics,
            "sent_metrics": sent_metrics,
            "ranked_docs": ranked_docs_json,
            "ranked_sents": ranked_sents_json
        })

    # average
    doc_avg = {}
    sent_avg= {}
    for k_ in K_VALUES:
        for m in ["P","R","F1","nDCG","MRR"]:
            arr_d = doc_storage[f"{m}@{k_}"]
            arr_s = sent_storage[f"{m}@{k_}"]
            doc_avg[f"{m}@{k_}"]  = float(np.mean(arr_d)) if len(arr_d)>0 else 0.0
            sent_avg[f"{m}@{k_}"] = float(np.mean(arr_s)) if len(arr_s)>0 else 0.0

    # save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if isinstance(o, (np.float32, np.int32)) else o)
    print(f"Saved eval results to {output_json}")

    print("=== DOC-LEVEL ===")
    for k_ in K_VALUES:
        print(f"K={k_} => P={doc_avg[f'P@{k_}']:.4f}, R={doc_avg[f'R@{k_}']:.4f}, "
              f"F1={doc_avg[f'F1@{k_}']:.4f}, nDCG={doc_avg[f'nDCG@{k_}']:.4f}, MRR={doc_avg[f'MRR@{k_}']:.4f}")
    print("=== SENT-LEVEL ===")
    for k_ in K_VALUES:
        print(f"K={k_} => P={sent_avg[f'P@{k_}']:.4f}, R={sent_avg[f'R@{k_}']:.4f}, "
              f"F1={sent_avg[f'F1@{k_}']:.4f}, nDCG={sent_avg[f'nDCG@{k_}']:.4f}, MRR={sent_avg[f'MRR@{k_}']:.4f}")

    return {"doc_avg": doc_avg, "sent_avg": sent_avg}

# ==============================
# 7. Main Function
# ==============================

def main():
    train_file = "train.json"
    val_file   = "val.json"
    test_file  = "test.json"
    output_dir = "outputs_coarse_fine"
    model_path = os.path.join(output_dir, "final_coarse_fine_model.pt")

    print("Loading LAIR-RAW data ...")
    train_data = load_lair_raw(train_file)
    val_data   = load_lair_raw(val_file)
    test_data  = load_lair_raw(test_file)

    # Setup base LLM + tokenizer + config + LoRA
    base_llm, tokenizer, base_config = setup_model_lora(
        model_name = MODEL_NAME,
        r=8,
        alpha=16,
        dropout=0.05,
        task_type=TaskType.FEATURE_EXTRACTION
    )

    # Automatically get the hidden_dim from the config (common fix for shape mismatch).
    hidden_dim = base_config.hidden_size
    print(f"Detected hidden_size = {hidden_dim}")

    coarse_fine_model = CoarseFineRetriever(hidden_dim=hidden_dim).to(device)

    if os.path.exists(model_path):
        print(f"Model checkpoint found at {model_path}. Skipping training.")
        coarse_fine_model.load_state_dict(torch.load(model_path))
    else:
        print("Start training (coarse-to-fine) ...")
        train_coarse_fine(
            model      = coarse_fine_model,
            base_llm   = base_llm,
            tokenizer  = tokenizer,
            train_data = train_data,
            num_epochs = 2,
            batch_size = 640,
            lr         = 1e-5,
            alpha_doc  = 1.0,
            alpha_sent = 1.0,
            output_dir = output_dir
        )

    print("Evaluating on val set ...")
    val_metrics = evaluate_coarse_fine(
        model     = coarse_fine_model,
        base_llm  = base_llm,
        tokenizer = tokenizer,
        eval_data = val_data,
        output_json="val_results.json"
    )
    print("Val metrics =>", val_metrics)

    print("Evaluating on test set ...")
    test_metrics = evaluate_coarse_fine(
        model     = coarse_fine_model,
        base_llm  = base_llm,
        tokenizer = tokenizer,
        eval_data = test_data,
        output_json="test_results.json"
    )
    print("Test metrics =>", test_metrics)

if __name__ == "__main__":
    main()


