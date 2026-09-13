import os
import re
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from transformers import (
    RobertaModel,
    DebertaV2Model,
    XLNetModel,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# LABELS (VERITE 3-way)
# ============================================================
LABEL_MAP = {"true": 0, "miscaptioned": 1, "out-of-context": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

SPECIAL_TOKENS = ["<CLAIM>", "<SUPPORT>", "<REFUTE>", "<END>"]
EXTERNAL_LABELS = {}  # claim_id -> raw label string


def normalize_label_text(raw: str) -> str:
    if raw is None:
        return "miscaptioned"
    t = str(raw).strip().lower()

    if t in {"true", "true.", "correct", "yes", "verified",
             "supported", "support", "support.", "entails", "entailment"}:
        return "true"

    if t in {"miscaptioned", "mis-captioned", "mis_captioned",
             "false", "false.", "fake", "incorrect",
             "refuted", "refute", "refute.", "contradiction"}:
        return "miscaptioned"

    if t in {"out-of-context", "out_of_context", "out of context",
             "context-shifted", "context shifted"}:
        return "out-of-context"

    if t in {"nei", "n.e.i", "not enough info", "not-enough-info",
             "unknown", "uncertain", "neutral", "unrelated"}:
        return "true"

    return "miscaptioned"


def clean_text(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_input_string_refute_only(claim: str, refute: str) -> str:
    # keep exact template; support is intentionally empty
    return f"<CLAIM> {claim} <SUPPORT>  <REFUTE> {refute} <END>"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_verite_labels(csv_path):
    """
    Uses CSV row index as claim_id (0..N-1), and reads column 'label' or 'veracity_label'.
    """
    mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in labels CSV: {csv_path}")

        lower_names = [c.lower() for c in reader.fieldnames]
        lab_col = None
        for cand in ["label", "veracity_label"]:
            if cand in lower_names:
                lab_col = reader.fieldnames[lower_names.index(cand)]
                break
        if lab_col is None:
            raise ValueError("Could not find label column (tried: label, veracity_label)")

        for i, row in enumerate(reader):
            lab = str(row.get(lab_col, "")).strip()
            if lab:
                mapping[str(i)] = lab

    print(f"✅ Loaded {len(mapping)} labels from {csv_path}")
    return mapping


# ============================================================
# MODEL CONFIGS (match training)
# ============================================================
MODEL_CONFIGS = {
    "roberta-large": {
        "model_class": RobertaModel,
        "max_position_embeddings": 512,
    },
    "microsoft/deberta-v3-large": {
        "model_class": DebertaV2Model,
        "max_position_embeddings": 512,
    },
    "xlnet-large-cased": {
        "model_class": XLNetModel,
        "max_position_embeddings": 1024,
    },
    "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit": {
        "model_class": AutoModel,
        "max_position_embeddings": 512,
    },
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit": {
        "model_class": AutoModel,
        "max_position_embeddings": 512,
    },
}


# ============================================================
# DATASET
# ============================================================
class FakeNewsDatasetRefuteOnly(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _pad_or_truncate(self, input_ids, attention_mask):
        if len(input_ids) >= self.max_len:
            return input_ids[:self.max_len], attention_mask[:self.max_len]
        pad_len = self.max_len - len(input_ids)
        input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]
        claim_id = str(item.get("claim_id", idx))
        claim = clean_text(item.get("claim_text", item.get("claim", "")))

        refute = item.get("refuting_text_evidence") or item.get("refute_justification")
        if refute is None:
            ref_list = item.get("refuting_text", [])
            if isinstance(ref_list, list):
                refute = " ".join(ref_list)
            else:
                refute = ref_list
        refute = clean_text(refute)

        input_text = build_input_string_refute_only(claim, refute)

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        input_ids, attention_mask = self._pad_or_truncate(input_ids, attention_mask)

        raw_label = item.get("label", None)
        if raw_label is None and EXTERNAL_LABELS:
            raw_label = EXTERNAL_LABELS.get(claim_id, None)
        if raw_label is None:
            raw_label = "miscaptioned"

        canonical = normalize_label_text(raw_label)
        label_idx = LABEL_MAP[canonical]

        return {
            "claim_id": claim_id,
            "claim": claim,
            "refute_justification": refute,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


# ============================================================
# CLASSIFIER (same as training: masked mean pooling + linear)
# ============================================================
class MeanPoolClassifier(nn.Module):
    def __init__(self, backbone, model_name: str, num_labels: int):
        super().__init__()
        self.backbone = backbone
        self.model_name = model_name

        hidden = getattr(self.backbone.config, "hidden_size", None)
        if hidden is None:
            raise ValueError(f"Could not determine hidden_size for model {model_name}")

        p0 = next(self.backbone.parameters())
        self.classifier = nn.Linear(hidden, num_labels).to(device=p0.device, dtype=p0.dtype)

    def forward(self, input_ids, attention_mask):
        if "xlnet" in self.model_name.lower():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last = out.hidden_states[-1]
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state

        am = attention_mask.to(last.dtype)
        pooled = (last * am.unsqueeze(-1)).sum(1) / (am.sum(1, keepdim=True) + 1e-9)
        logits = self.classifier(pooled)
        return logits


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def _get_expected_vocab_from_adapter(checkpoint_dir: str):
    """
    Reads expected vocab size from adapter model weights if embeddings were saved.
    """
    st_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if os.path.isfile(st_path):
        from safetensors.torch import load_file
        sd = load_file(st_path)
        for k, v in sd.items():
            if k.endswith("embeddings.word_embeddings.weight"):
                return v.shape[0]
        return None

    bin_path = os.path.join(checkpoint_dir, "adapter_model.bin")
    if os.path.isfile(bin_path):
        sd = torch.load(bin_path, map_location="cpu")
        for k, v in sd.items():
            if k.endswith("embeddings.word_embeddings.weight"):
                return v.shape[0]
        return None

    return None


def load_classifier_head(checkpoint_dir: str, device):
    head_path = os.path.join(checkpoint_dir, "classifier_head.pth")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"Missing classifier_head.pth in {checkpoint_dir}")
    ckpt = torch.load(head_path, map_location=device)
    return ckpt["classifier"]


def load_backbone_and_tokenizer(model_name: str, checkpoint_dir: str, force_single_gpu: bool):
    """
    IMPORTANT FIX:
      - tokenizer must be loaded from base model_name (checkpoint_dir doesn't have vocab files)
      - add SPECIAL_TOKENS exactly like training
      - resize embeddings to match adapter vocab size
      - then load PEFT adapter
    """
    # ✅ load tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # ensure pad token
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # add special tokens exactly like training
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    # determine what vocab size adapter expects
    expected_vocab = _get_expected_vocab_from_adapter(checkpoint_dir)
    if expected_vocab is None:
        expected_vocab = len(tokenizer)
    else:
        # if tokenizer is smaller, add dummy tokens to reach adapter vocab size
        if len(tokenizer) < expected_vocab:
            need = expected_vocab - len(tokenizer)
            extra = [f"<EXTRA_{i}>" for i in range(need)]
            tokenizer.add_tokens(extra)

        # if tokenizer is larger, something is inconsistent; fail fast
        if len(tokenizer) > expected_vocab:
            raise RuntimeError(
                f"Tokenizer vocab ({len(tokenizer)}) > adapter expected vocab ({expected_vocab}). "
                f"This usually means tokenizer mismatch vs training."
            )

    device_map = {"": 0} if force_single_gpu else "auto"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    backbone = MODEL_CONFIGS[model_name]["model_class"].from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16
    )

    # resize embeddings to tokenizer length (which now matches adapter expected vocab)
    cur_vocab = backbone.get_input_embeddings().weight.shape[0]
    if cur_vocab != len(tokenizer):
        print(f"Resizing embeddings: {cur_vocab} -> {len(tokenizer)}")
        backbone.resize_token_embeddings(len(tokenizer))

    # load adapter
    backbone = PeftModel.from_pretrained(backbone, checkpoint_dir, is_trainable=False)
    backbone.eval()

    return backbone, tokenizer


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--force_single_gpu", action="store_true")
    args = parser.parse_args()

    global EXTERNAL_LABELS
    EXTERNAL_LABELS = {}
    if args.labels_csv:
        EXTERNAL_LABELS = load_verite_labels(args.labels_csv)

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_name '{args.model_name}'. Options: {list(MODEL_CONFIGS.keys())}")

    print(f"Loading model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_dir}")

    backbone, tokenizer = load_backbone_and_tokenizer(
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        force_single_gpu=args.force_single_gpu
    )

    model_device = next(backbone.parameters()).device
    print(f"✅ Backbone device: {model_device}")

    clf = MeanPoolClassifier(backbone, args.model_name, num_labels=len(LABEL_MAP))

    # load classifier head trained in your script
    head_sd = load_classifier_head(args.checkpoint_dir, device=model_device)
    clf.classifier.load_state_dict(head_sd)
    clf.eval()

    data = load_json(args.test_json)
    max_len = MODEL_CONFIGS[args.model_name]["max_position_embeddings"]

    dataset = FakeNewsDatasetRefuteOnly(data, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset))

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(model_device)
            attention_mask = batch["attention_mask"].to(model_device)

            logits = clf(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            for i in range(len(preds)):
                true_idx = int(batch["label"][i].item())
                pred_idx = int(preds[i])

                rows.append({
                    "claim_id": batch["claim_id"][i],
                    "claim": batch["claim"][i],
                    "refute_justification": batch["refute_justification"][i],
                    "true_label": INV_LABEL_MAP[true_idx],
                    "pred_label": INV_LABEL_MAP[pred_idx],
                    "prob_true": float(probs[i][LABEL_MAP["true"]]),
                    "prob_miscaptioned": float(probs[i][LABEL_MAP["miscaptioned"]]),
                    "prob_out_of_context": float(probs[i][LABEL_MAP["out-of-context"]]),
                })

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Saved predictions CSV: {args.output_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

# import os
# import re
# import json
# import csv
# import argparse
# from typing import Dict, Any, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm

# from sklearn.metrics import classification_report, accuracy_score, f1_score

# from transformers import CLIPTokenizer, CLIPTextModel


# # ============================================================
# # LABELS (VERITE 3-way)
# # ============================================================
# LABEL_MAP = {"true": 0, "miscaptioned": 1, "out-of-context": 2}
# INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# SPECIAL_TOKENS = ["<CLAIM>", "<SUPPORT>", "<REFUTE>", "<END>"]


# def normalize_label_text(raw: str) -> str:
#     if raw is None:
#         return "miscaptioned"
#     t = str(raw).strip().lower()

#     if t in {
#         "true", "true.", "correct", "yes", "verified",
#         "supported", "support", "support.", "entails", "entailment"
#     }:
#         return "true"

#     if t in {
#         "miscaptioned", "mis-captioned", "mis_captioned",
#         "false", "false.", "fake", "incorrect",
#         "refuted", "refute", "refute.", "contradiction"
#     }:
#         return "miscaptioned"

#     if t in {
#         "out-of-context", "out_of_context", "out of context",
#         "context-shifted", "context shifted"
#     }:
#         return "out-of-context"

#     if t in {
#         "nei", "n.e.i", "not enough info", "not-enough-info",
#         "unknown", "uncertain", "neutral", "unrelated"
#     }:
#         return "true"

#     return "miscaptioned"


# def clean_text(x: Any) -> str:
#     if x is None:
#         return ""
#     return re.sub(r"\s+", " ", str(x)).strip()


# def _join_if_list(x: Any) -> str:
#     if x is None:
#         return ""
#     if isinstance(x, list):
#         return " ".join([clean_text(t) for t in x if t is not None])
#     return clean_text(x)


# def build_input_string(claim: str, support: str, refute: str, refute_only: bool) -> str:
#     # Keep token pattern stable across runs (support empty if refute-only)
#     if refute_only:
#         support = ""
#     return f"<CLAIM> {claim} <SUPPORT> {support} <REFUTE> {refute} <END>"


# def load_json(path: str) -> List[Dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def load_verite_labels_csv(csv_path: str) -> Dict[str, str]:
#     """
#     Uses CSV row index as claim_id (0..N-1), reads column 'label' or 'veracity_label'.
#     """
#     mapping: Dict[str, str] = {}
#     with open(csv_path, "r", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         if reader.fieldnames is None:
#             raise ValueError(f"No header found in labels CSV: {csv_path}")

#         lower_names = [c.lower() for c in reader.fieldnames]
#         lab_col = None
#         for cand in ["label", "veracity_label"]:
#             if cand in lower_names:
#                 lab_col = reader.fieldnames[lower_names.index(cand)]
#                 break
#         if lab_col is None:
#             raise ValueError("Could not find label column (tried: label, veracity_label)")

#         for i, row in enumerate(reader):
#             lab = str(row.get(lab_col, "")).strip()
#             if lab:
#                 mapping[str(i)] = lab

#     print(f"✅ Loaded {len(mapping)} labels from {csv_path}")
#     return mapping


# # ============================================================
# # BASELINE HEAD (FIXED to match reddot_verite_3class_best.pt)
# #   - cls_token in checkpoint is shape [512] (NOT [1,1,512])
# #   - verdict MLP in checkpoint is 512 -> 256 -> 3
# # ============================================================
# class VerdictClassifier(nn.Module):
#     def __init__(self, in_dim: int = 512, hidden_dim: int = 256, num_labels: int = 3):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(in_dim)
#         self.fc = nn.Linear(in_dim, hidden_dim)          # 512 -> 256
#         self.output_layer = nn.Linear(hidden_dim, num_labels)  # 256 -> 3

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.layer_norm(x)
#         x = F.gelu(self.fc(x))
#         return self.output_layer(x)


# class RedDotBaselineHead(nn.Module):
#     """
#     Expects token embeddings: (B, S, 512) + attention_mask: (B, S) with 1=valid, 0=pad
#     Produces logits: (B, 3)
#     """
#     def __init__(
#         self,
#         d_model: int = 512,
#         nhead: int = 8,
#         ff: int = 2048,
#         num_layers: int = 4,
#         num_labels: int = 3
#     ):
#         super().__init__()

#         # IMPORTANT: must match checkpoint (512,)
#         self.cls_token = nn.Parameter(torch.zeros(d_model))  # (512,)

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=ff,
#             activation="gelu",
#             batch_first=True,
#         )
#         self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

#         # IMPORTANT: must match checkpoint 512 -> 256 -> 3
#         self.verdict_classifier = VerdictClassifier(in_dim=d_model, hidden_dim=256, num_labels=num_labels)

#     def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         # x: (B,S,D), attention_mask: (B,S) with 1 valid, 0 pad
#         B, S, D = x.shape

#         cls = self.cls_token.view(1, 1, D).expand(B, 1, D)  # (B,1,D)
#         x = torch.cat([cls, x], dim=1)  # (B,S+1,D)

#         attention_mask = torch.cat(
#             [torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype), attention_mask],
#             dim=1
#         )  # (B,S+1)

#         key_padding_mask = (attention_mask == 0)  # True where pad
#         h = self.transformer(x, src_key_padding_mask=key_padding_mask)
#         cls_h = h[:, 0, :]  # (B,D)
#         logits = self.verdict_classifier(cls_h)
#         return logits


# # ============================================================
# # CHECKPOINT LOADER (robust)
# # ============================================================
# def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
#     """
#     Accepts:
#       - a raw state_dict (all values are tensors)
#       - a checkpoint dict with common keys: state_dict/model/model_state_dict
#     """
#     if isinstance(obj, dict):
#         # Case 1: already a state_dict
#         if len(obj) > 0 and all(isinstance(v, torch.Tensor) for v in obj.values()):
#             return obj

#         # Case 2: wrapped checkpoint
#         for k in ["state_dict", "model_state_dict", "model", "net", "head_state_dict"]:
#             if k in obj and isinstance(obj[k], dict) and len(obj[k]) > 0:
#                 inner = obj[k]
#                 if all(isinstance(v, torch.Tensor) for v in inner.values()):
#                     return inner

#     raise ValueError(
#         "Could not extract a valid state_dict from the checkpoint. "
#         "Print the keys of torch.load(...) to see structure."
#     )


# def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
#     out = {}
#     for k, v in state_dict.items():
#         nk = k
#         for p in prefixes:
#             if nk.startswith(p):
#                 nk = nk[len(p):]
#         out[nk] = v
#     return out


# # ============================================================
# # PREDICTION
# # ============================================================
# @torch.no_grad()
# def run_prediction(
#     data: List[Dict[str, Any]],
#     labels_map: Dict[str, str],
#     clip_tokenizer: CLIPTokenizer,
#     clip_text_model: CLIPTextModel,
#     head: RedDotBaselineHead,
#     device: torch.device,
#     batch_size: int,
#     max_length: int,
#     refute_only: bool,
# ) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
#     rows: List[Dict[str, Any]] = []
#     y_true: List[int] = []
#     y_pred: List[int] = []

#     for start in tqdm(range(0, len(data), batch_size), desc="Predicting"):
#         batch_items = data[start:start + batch_size]

#         claim_ids: List[str] = []
#         claims: List[str] = []
#         supports: List[str] = []
#         refutes: List[str] = []
#         true_idx_list: List[int] = []
#         texts: List[str] = []

#         for i, item in enumerate(batch_items):
#             cid = str(item.get("claim_id", start + i))
#             claim = clean_text(item.get("claim_text", item.get("claim", "")))

#             sup = (
#                 item.get("supporting_text_evidence")
#                 or item.get("support_justification")
#                 or _join_if_list(item.get("supporting_text", ""))
#             )
#             sup = clean_text(sup)

#             ref = (
#                 item.get("refuting_text_evidence")
#                 or item.get("refute_justification")
#                 or _join_if_list(item.get("refuting_text", ""))
#             )
#             ref = clean_text(ref)

#             raw_label = item.get("label", None)
#             if raw_label is None and labels_map:
#                 raw_label = labels_map.get(cid, None)
#             if raw_label is None:
#                 raw_label = "miscaptioned"

#             canonical = normalize_label_text(raw_label)
#             true_idx = LABEL_MAP[canonical]

#             inp = build_input_string(claim, sup, ref, refute_only=refute_only)

#             claim_ids.append(cid)
#             claims.append(claim)
#             supports.append(sup)
#             refutes.append(ref)
#             true_idx_list.append(true_idx)
#             texts.append(inp)

#         tok = clip_tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt",
#         )

#         # move to device
#         tok = {k: v.to(device) for k, v in tok.items()}

#         clip_out = clip_text_model(**tok)
#         token_embeds = clip_out.last_hidden_state  # (B,S,512)
#         attn_mask = tok["attention_mask"]          # (B,S)

#         logits = head(token_embeds, attn_mask)     # (B,3)
#         probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
#         preds = probs.argmax(axis=1).tolist()

#         for i in range(len(preds)):
#             ti = int(true_idx_list[i])
#             pi = int(preds[i])
#             y_true.append(ti)
#             y_pred.append(pi)

#             rows.append({
#                 "claim_id": claim_ids[i],
#                 "claim": claims[i],
#                 "support_justification": supports[i],
#                 "refute_justification": refutes[i],
#                 "true_label": INV_LABEL_MAP[ti],
#                 "pred_label": INV_LABEL_MAP[pi],
#                 "prob_true": float(probs[i][LABEL_MAP["true"]]),
#                 "prob_miscaptioned": float(probs[i][LABEL_MAP["miscaptioned"]]),
#                 "prob_out_of_context": float(probs[i][LABEL_MAP["out-of-context"]]),
#             })

#     return rows, y_true, y_pred


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test_json", type=str, required=True)
#     parser.add_argument("--baseline_pt", type=str, required=True, help="Path to reddot_verite_3class_best.pt")
#     parser.add_argument("--output_csv", type=str, required=True)
#     parser.add_argument("--output_report", type=str, default=None, help="Optional path to save classification report txt")
#     parser.add_argument("--labels_csv", type=str, default=None, help="Optional VERITE.csv to get true labels by claim_id (row index)")
#     parser.add_argument("--batch_size", type=int, default=16)

#     # text encoder used to create 512-d token embeddings
#     parser.add_argument("--clip_text_encoder", type=str, default="openai/clip-vit-base-patch32")

#     # CLIP max length (77 is standard for CLIP)
#     parser.add_argument("--max_length", type=int, default=77)

#     # if your baseline was trained with only refute justification
#     parser.add_argument("--refute_only", action="store_true")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # labels map (optional)
#     labels_map: Dict[str, str] = {}
#     if args.labels_csv:
#         labels_map = load_verite_labels_csv(args.labels_csv)

#     # load data
#     data = load_json(args.test_json)
#     print(f"✅ Loaded test samples: {len(data)}")

#     # load CLIP tokenizer + text model
#     clip_tokenizer = CLIPTokenizer.from_pretrained(args.clip_text_encoder)
#     clip_text_model = CLIPTextModel.from_pretrained(args.clip_text_encoder).to(device)
#     clip_text_model.eval()

#     # load baseline head (MUST match checkpoint architecture)
#     head = RedDotBaselineHead(d_model=512, nhead=8, ff=2048, num_layers=4, num_labels=3).to(device)
#     ckpt_obj = torch.load(args.baseline_pt, map_location="cpu")
#     sd = _extract_state_dict(ckpt_obj)

#     # strip common wrappers
#     sd = _strip_prefix(sd, prefixes=["module.", "head.", "baseline_head.", "model."])

#     # now this should load cleanly if architecture matches
#     missing, unexpected = head.load_state_dict(sd, strict=False)
#     if missing or unexpected:
#         print("⚠️ State dict load notes:")
#         if missing:
#             print("  Missing keys:", missing)
#         if unexpected:
#             print("  Unexpected keys:", unexpected)

#     # If you want strict loading (recommended), enable this after confirming keys:
#     # head.load_state_dict(sd, strict=True)

#     head.eval()

#     # predict
#     rows, y_true, y_pred = run_prediction(
#         data=data,
#         labels_map=labels_map,
#         clip_tokenizer=clip_tokenizer,
#         clip_text_model=clip_text_model,
#         head=head,
#         device=device,
#         batch_size=args.batch_size,
#         max_length=args.max_length,
#         refute_only=args.refute_only,
#     )

#     # save csv
#     os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
#     if not rows:
#         raise RuntimeError("No rows produced. Check your input JSON formatting.")
#     with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
#         writer.writeheader()
#         writer.writerows(rows)
#     print(f"✅ Saved predictions CSV: {args.output_csv}")
#     print(f"Rows written: {len(rows)}")

#     # classification report (meaningful if you have true labels)
#     if len(y_true) == len(y_pred) and len(y_true) > 0:
#         labels = [0, 1, 2]
#         target_names = [INV_LABEL_MAP[i] for i in labels]
#         report = classification_report(
#             y_true, y_pred,
#             labels=labels,
#             target_names=target_names,
#             digits=4,
#             zero_division=0
#         )
#         acc = accuracy_score(y_true, y_pred)
#         mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

#         print("\n" + "=" * 80)
#         print("FULL CLASSIFICATION REPORT (Baseline .pt on given test_json)")
#         print("=" * 80)
#         print(report)
#         print(f"Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")

#         if args.output_report:
#             os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
#             with open(args.output_report, "w", encoding="utf-8") as f:
#                 f.write(report + "\n")
#                 f.write(f"\nAccuracy: {acc:.6f}\nMacro-F1: {mf1:.6f}\n")
#             print(f"✅ Saved report: {args.output_report}")
#     else:
#         print("⚠️ Skipped report (no aligned true labels). Provide --labels_csv or ensure JSON has 'label'.")


# if __name__ == "__main__":
#     main()
