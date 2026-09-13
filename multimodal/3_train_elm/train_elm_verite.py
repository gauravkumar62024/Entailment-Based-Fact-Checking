import json
import torch
import copy
import numpy as np
import random
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, RobertaModel,
    DebertaV2Tokenizer, DebertaV2Model,
    XLNetTokenizer, XLNetModel,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
from collections import Counter
import csv

# Maps claim_id (string) -> raw label string ('true', 'miscaptioned', 'out-of-context')
EXTERNAL_LABELS = {}


# ✅ Output directory
output_dir = "./checkpoints"
os.makedirs(output_dir, exist_ok=True)

# ======================================================================
# LABELS / VERITE
# ======================================================================

# VERITE 3-way label space
# 0 -> "true"
# 1 -> "miscaptioned"
# 2 -> "out-of-context"
LABEL_MAP = {
    "true": 0,
    "miscaptioned": 1,
    "out-of-context": 2,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def normalize_label_text(raw: str) -> str:
    """
    Map various label strings (VERITE / generic) into
    canonical VERITE keys used in LABEL_MAP:
        'true', 'miscaptioned', 'out-of-context'
    """
    if raw is None:
        return "miscaptioned"

    t = str(raw).strip().lower()

    # ---- TRUE-LIKE ----
    if t in {
        "true", "true.", "correct", "yes", "verified",
        "supported", "support", "support.", "entails", "entailment"
    }:
        return "true"

    # ---- MISCAPTIONED-LIKE (generic refuted / false) ----
    if t in {
        "miscaptioned", "mis-captioned", "mis_captioned",
        "false", "false.", "fake", "incorrect",
        "refuted", "refute", "refute.", "contradiction"
    }:
        return "miscaptioned"

    # ---- OUT-OF-CONTEXT-LIKE ----
    if t in {
        "out-of-context", "out_of_context", "out of context",
        "context-shifted", "context shifted"
    }:
        return "out-of-context"

    # ---- NEI / unknown / neutral – map to 'true' or 'miscaptioned' depending on your design
    # Here we map NEI to 'true' only if you consider them truthful-but-insufficient;
    # otherwise you can change to 'miscaptioned'.
    if t in {
        "nei", "n.e.i", "not enough info", "not-enough-info",
        "unknown", "uncertain", "neutral", "unrelated"
    }:
        # choose one; here defaulting to "true" is arbitrary.
        return "true"

    # Default fall-back
    return "miscaptioned"


# Special tokens
SPECIAL_TOKENS = ["<CLAIM>", "<SUPPORT>", "<REFUTE>", "<END>"]

MODEL_CONFIGS = {
    "roberta-large": {
        "model_class": RobertaModel,
        "tokenizer_class": RobertaTokenizer,
        "max_position_embeddings": 512,
        "hidden_size": 1024,
    },
    "microsoft/deberta-v3-large": {
        "model_class": DebertaV2Model,
        "tokenizer_class": DebertaV2Tokenizer,
        "max_position_embeddings": 512,
        "hidden_size": 1024,
    },
    "xlnet-large-cased": {
        "model_class": XLNetModel,
        "tokenizer_class": XLNetTokenizer,
        "max_position_embeddings": 1024,
        "hidden_size": 1024,
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Early stopping handler class with minimum epoch requirement"""
    def __init__(self, patience=3, min_delta=0, mode='max', min_epochs=12, min_f1=0.79):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs  # Minimum epochs to run
        self.min_f1 = min_f1  # Minimum F1 score requirement
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.current_epoch = 0

    def __call__(self, score, current_epoch):
        self.current_epoch = current_epoch

        # ✅ Don't stop before minimum epochs if F1 is below threshold
        if current_epoch < self.min_epochs and score < self.min_f1:
            print(f"⚠️  Early stopping suspended: epoch {current_epoch} < {self.min_epochs} and F1 {score:.4f} < {self.min_f1}")
            return True  # Still save models, but don't stop

        if self.best_score is None:
            self.best_score = score
            return True  # Signal to save first model

        if self.mode == 'max':
            improvement = score > (self.best_score + self.min_delta)
        else:
            improvement = score < (self.best_score - self.min_delta)

        if improvement:
            self.best_score = score
            self.counter = 0
            return True  # Signal to save improved model
        else:
            self.counter += 1
            # ✅ Only consider early stopping after minimum epochs OR if F1 meets requirement
            if self.counter >= self.patience and (current_epoch >= self.min_epochs or score >= self.min_f1):
                self.early_stop = True
                print(f"🚨 Early stopping triggered: {self.counter} epochs without improvement")
            return False  # No save


def load_data(json_path):
    """Load dataset from JSON file (Factify / VERITE / others)"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file {json_path} not found")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} samples from {json_path}")
    return data


def clean_text(text):
    """Clean text by removing extra whitespace"""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def build_input_string(claim: str, support: str, refute: str) -> str:
    return f"<CLAIM> {claim} <SUPPORT> {support} <REFUTE> {refute} <END>"

def load_verite_labels(csv_path):
    """
    Load VERITE gold labels from a CSV *without* touching the JSON files.

    Your CSV has columns: 'Unnamed: 0', 'caption', 'image_path', 'label'

    We will:
      - treat the CSV ROW INDEX (0, 1, 2, ...) as the claim_id
      - map: str(row_index) -> label

    This matches how claim_id was created in your VERITE JSONs
    (using 'idx' from df.iterrows()).
    """
    mapping = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in labels CSV: {csv_path}")

        # Find the label column (case-insensitive)
        lower_names = [c.lower() for c in reader.fieldnames]
        lab_col = None
        for cand in ["label", "veracity_label"]:
            if cand in lower_names:
                lab_col = reader.fieldnames[lower_names.index(cand)]
                break
        if lab_col is None:
            raise ValueError(
                "Could not find a label column in VERITE labels CSV "
                "(tried: label, veracity_label)"
            )

        # Enumerate rows: index i becomes the claim_id
        for i, row in enumerate(reader):
            lab = str(row.get(lab_col, "")).strip()
            if not lab:
                continue
            cid = str(i)        # <- this is the claim_id used in JSON
            mapping[cid] = lab

    print(f"✅ Loaded {len(mapping)} labels from {csv_path}")
    return mapping




class FakeNewsDataset(Dataset):
    """
    Custom dataset with intelligent truncation.

    Compatible with:
      - Original Factify-style JSON (label, supporting_text_evidence, refuting_text_evidence)
      - VERITE JSON from your VLM pipeline (no label fields), using EXTERNAL_LABELS[claim_id]
    """
    def __init__(self, data, tokenizer, max_len=512, truncation_strategy="head_tail"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation_strategy = truncation_strategy

    def __len__(self):
        return len(self.data)

    def _truncate_smart(self, input_ids, attention_mask, special_token_ids):
        """Smart truncation that preserves special tokens and keeps head+tail"""
        actual_len = len(input_ids)
        if actual_len <= self.max_len:
            pad_len = self.max_len - actual_len
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
            return input_ids, attention_mask

        # Find special token positions
        special_positions = []
        for token_id in special_token_ids:
            positions = (input_ids == token_id).nonzero(as_tuple=False)
            if len(positions) > 0:
                special_positions.append(int(positions[0].item()))

        if not special_positions:
            truncated = input_ids[:self.max_len]
            trunc_mask = attention_mask[:self.max_len]
            return truncated, trunc_mask

        special_end = max(special_positions) + 1
        tokens_after_special = actual_len - special_end

        if special_end >= self.max_len:
            truncated = input_ids[:self.max_len]
            trunc_mask = attention_mask[:self.max_len]
            return truncated, trunc_mask

        remaining_space = self.max_len - special_end
        tail_size = min(remaining_space, tokens_after_special // 2)

        head = input_ids[:special_end]
        tail = input_ids[-(tail_size):] if tail_size > 0 else torch.tensor([], dtype=input_ids.dtype)

        combined = torch.cat([head, tail])[:self.max_len]
        new_attention = torch.ones(len(combined), dtype=attention_mask.dtype)

        pad_len = self.max_len - len(combined)
        if pad_len > 0:
            combined = torch.cat([combined, torch.zeros(pad_len, dtype=combined.dtype)])
            new_attention = torch.cat([new_attention, torch.zeros(pad_len, dtype=new_attention.dtype)])

        return combined, new_attention

    def __getitem__(self, idx):
        item = self.data[idx]

        # ---- CLAIM ----
        claim = clean_text(item.get("claim_text", item.get("claim", "")))

        # ---- SUPPORT JUSTIFICATION ----
        supporting_justification = (
            item.get("supporting_text_evidence")
            or item.get("support_justification")
        )
        if supporting_justification is None:
            supp_list = item.get("supporting_text", [])
            if isinstance(supp_list, list):
                supporting_justification = " ".join(supp_list)
            else:
                supporting_justification = supp_list

        supporting_justification = clean_text(supporting_justification)

        # ---- REFUTE JUSTIFICATION ----
        refuting_justification = (
            item.get("refuting_text_evidence")
            or item.get("refute_justification")
        )
        if refuting_justification is None:
            ref_list = item.get("refuting_text", [])
            if isinstance(ref_list, list):
                refuting_justification = " ".join(ref_list)
            else:
                refuting_justification = ref_list

        refuting_justification = clean_text(refuting_justification)

        # Build final input string
        input_text = build_input_string(claim, supporting_justification, refuting_justification)

        encoding = self.tokenizer(
            input_text,
            max_length=None,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        tid = self.tokenizer.convert_tokens_to_ids
        special_token_ids = [tid("<CLAIM>"), tid("<SUPPORT>"), tid("<REFUTE>"), tid("<END>")]

        input_ids, attention_mask = self._truncate_smart(input_ids, attention_mask, special_token_ids)

        actual_len = int(attention_mask.sum().item())
        actual_len = max(5, min(actual_len, self.max_len))

        t_claim, t_support, t_refute, t_end = special_token_ids
        pos_claim = (input_ids == t_claim).nonzero(as_tuple=False)
        pos_support = (input_ids == t_support).nonzero(as_tuple=False)
        pos_refute = (input_ids == t_refute).nonzero(as_tuple=False)
        pos_end = (input_ids == t_end).nonzero(as_tuple=False)

        c_idx = int(pos_claim[0].item()) if len(pos_claim) > 0 else 1
        s_idx = int(pos_support[0].item()) if len(pos_support) > 0 else min(c_idx + 2, actual_len - 3)
        r_idx = int(pos_refute[0].item()) if len(pos_refute) > 0 else min(s_idx + 2, actual_len - 2)
        e_idx = int(pos_end[0].item()) if len(pos_end) > 0 else actual_len - 1

        c_idx = max(0, min(c_idx, actual_len - 4))
        s_idx = max(c_idx + 1, min(s_idx, actual_len - 3))
        r_idx = max(s_idx + 1, min(r_idx, actual_len - 2))
        e_idx = max(r_idx + 1, min(e_idx, actual_len - 1))

        # ---- LABEL LOOKUP ----
        # Priority:
        #   1) item["label"]          (if present in JSON)
        #   2) EXTERNAL_LABELS[claim_id] (loaded from VERITE CSV)
        #   3) default "miscaptioned" (if nothing else is available)
        raw_label = item.get("label", None)

        if raw_label is None and EXTERNAL_LABELS:
            claim_id = str(item.get("claim_id", "")).strip()
            if claim_id in EXTERNAL_LABELS:
                raw_label = EXTERNAL_LABELS[claim_id]

        if raw_label is None:
            raw_label = "miscaptioned"  # last-resort default

        canonical = normalize_label_text(raw_label)  # "true" / "miscaptioned" / "out-of-context"
        label_idx = LABEL_MAP[canonical]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "c_idx": torch.tensor(c_idx, dtype=torch.long),
            "s_idx": torch.tensor(s_idx, dtype=torch.long),
            "r_idx": torch.tensor(r_idx, dtype=torch.long),
            "e_idx": torch.tensor(e_idx, dtype=torch.long)
        }



def eval_model(model, dataloader, device, phase="Validation"):
    """Evaluate the model on a given dataloader"""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            c_idx = batch["c_idx"].to(device)
            s_idx = batch["s_idx"].to(device)
            r_idx = batch["r_idx"].to(device)
            e_idx = batch["e_idx"].to(device)

            outputs = model(input_ids, attention_mask, c_idx, s_idx, r_idx, e_idx)
            preds = torch.argmax(outputs["logits"], dim=1)

            # ✅ Handle bfloat16 dtype before converting to numpy
            if preds.dtype == torch.bfloat16:
                preds = preds.float()
            if labels.dtype == torch.bfloat16:
                labels = labels.float()

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall = recall_score(true_labels, predictions, average="macro", zero_division=0)

    print(f"{phase} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    if phase == "Test":
        # Force 3-class VERITE label space even if only 1–2 classes appear in y
        label_indices = list(range(len(LABEL_MAP)))      # [0, 1, 2]
        target_names = [INV_LABEL_MAP[i] for i in label_indices]

        report = classification_report(
            true_labels,
            predictions,
            labels=label_indices,       # <-- important change
            target_names=target_names,  # must match len(labels)
            digits=4,
            zero_division=0
        )

        print("\nDetailed Classification Report:")
        print(report)

    return f1


def compute_class_weights(train_data):
    """
    Compute class weights for balanced training.

    Uses:
      - item["label"]               (if present in JSON)
      - EXTERNAL_LABELS[claim_id]   (if provided from VERITE CSV)
    All labels are mapped into VERITE label space via normalize_label_text.
    """
    labels = []

    for item in train_data:
        raw = item.get("label", None)

        # If JSON lacks label, try external mapping
        if raw is None and EXTERNAL_LABELS:
            cid = str(item.get("claim_id", "")).strip()
            raw = EXTERNAL_LABELS.get(cid, None)

        # Still nothing? we skip this example for class-weight computation
        if raw is None:
            continue

        canonical = normalize_label_text(raw)
        if canonical not in LABEL_MAP:
            continue

        labels.append(LABEL_MAP[canonical])

    if not labels:
        print("⚠️ No valid labels found in training data for class weights. Returning None.")
        return None

    # Debug: show distribution
    label_names = [INV_LABEL_MAP[i] for i in labels]
    print("Label distribution in TRAIN:", Counter(label_names))

    labels = np.array(labels, dtype=int)
    unique_classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )

    # Expand to full 3-class vector [true, miscaptioned, out-of-context]
    full_weights = np.ones(len(LABEL_MAP), dtype=np.float32)
    for c, w in zip(unique_classes, weights):
        full_weights[c] = w

    return torch.tensor(full_weights, dtype=torch.float)




class FusedClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, special_tokens: list, use_qlora: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_qlora = use_qlora

        config = MODEL_CONFIGS[model_name]

        if use_qlora:
            print(f"Loading {model_name} with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.backbone = config["model_class"].from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            device = next(self.backbone.parameters()).device
            dtype = torch.bfloat16
        else:
            self.backbone = config["model_class"].from_pretrained(model_name)
            device = torch.device("cpu")
            dtype = torch.float32

        hidden = config["hidden_size"]

        self.classifier = nn.Linear(hidden, num_labels, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden, hidden, dtype=dtype, device=device)

        self.special_tokens = special_tokens

    def forward(self, input_ids, attention_mask, c_idx, s_idx, r_idx, e_idx, labels=None):
        B, L = input_ids.shape

        if "xlnet" in self.model_name.lower():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last = outputs.hidden_states[-1]
        else:
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last = outputs.last_hidden_state

        attention_mask = attention_mask.to(last.dtype)

        # CLS pooling (actually masked mean pooling)
        cls_pool = (last * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(1, keepdim=True) + 1e-9)
        logits = self.classifier(cls_pool)

        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)

        return {"logits": logits, "loss": ce_loss, "ce": ce_loss}


def save_model_checkpoint(model, optimizer, epoch, val_f1, checkpoint_dir, use_qlora):
    """Proper saving for QLoRA models"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    if use_qlora:
        if isinstance(model.backbone, PeftModel):
            model.backbone.save_pretrained(checkpoint_dir)
        else:
            torch.save(model.backbone.state_dict(), os.path.join(checkpoint_dir, "backbone.pth"))
        print(f"✅ Saved LoRA adapters to {checkpoint_dir}")
    else:
        torch.save(model.backbone.state_dict(), os.path.join(checkpoint_dir, "backbone.pth"))
        print(f"✅ Saved full backbone to {checkpoint_dir}")

    torch.save({
        'classifier': model.classifier.state_dict(),
        'epoch': epoch,
        'val_f1': val_f1,
    }, os.path.join(checkpoint_dir, "classifier_head.pth"))

    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, "optimizer.pth"))

    print(f"✅ Checkpoint saved at epoch {epoch} with val_f1: {val_f1:.4f}")


def load_model_checkpoint(model, optimizer, checkpoint_dir, device):
    """Proper loading for QLoRA models"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")

    heads_path = os.path.join(checkpoint_dir, "classifier_head.pth")
    if os.path.exists(heads_path):
        checkpoint = torch.load(heads_path, map_location=device)
        model.classifier.load_state_dict(checkpoint['classifier'])
        print(f"✅ Loaded classifier head from epoch {checkpoint['epoch']} with val_f1: {checkpoint['val_f1']:.4f}")
    else:
        raise FileNotFoundError(f"Classifier head file {heads_path} not found")

    if model.use_qlora:
        if not isinstance(model.backbone, PeftModel):
            try:
                model.backbone = PeftModel.from_pretrained(
                    model.backbone,
                    checkpoint_dir,
                    is_trainable=True
                )
                print(f"✅ Loaded LoRA adapters from {checkpoint_dir}")
            except Exception as e:
                print(f"⚠️  Could not load LoRA adapters: {e}")
                print("Continuing with current model state...")
    else:
        backbone_path = os.path.join(checkpoint_dir, "backbone.pth")
        if os.path.exists(backbone_path):
            model.backbone.load_state_dict(torch.load(backbone_path, map_location=device))
            print(f"✅ Loaded full backbone from {backbone_path}")

    if optimizer is not None:
        opt_path = os.path.join(checkpoint_dir, "optimizer.pth")
        if os.path.exists(opt_path):
            opt_checkpoint = torch.load(opt_path, map_location=device)
            optimizer.load_state_dict(opt_checkpoint['optimizer_state_dict'])
            print(f"✅ Loaded optimizer state")

    return checkpoint.get('epoch', 0), checkpoint.get('val_f1', 0.0)


def train_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler,
                device, class_weights=None, epochs=20, patience=3, checkpoint_dir="best_model",
                min_epochs=12, min_f1=0.79):
    """Training with only CE loss"""
    early_stopping = EarlyStopping(patience=patience, min_epochs=min_epochs, min_f1=min_f1)
    best_val_f1 = 0

    if class_weights is not None:
        class_weights = class_weights.to(device)

    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{epochs}")
        loop = tqdm(train_dataloader, leave=True)
        total_loss = 0
        correct = 0
        total = 0

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            c_idx = batch["c_idx"].to(device)
            s_idx = batch["s_idx"].to(device)
            r_idx = batch["r_idx"].to(device)
            e_idx = batch["e_idx"].to(device)

            outputs = model(input_ids, attention_mask, c_idx, s_idx, r_idx, e_idx, labels=labels)
            loss = outputs["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs["logits"], dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item(), accuracy=correct / total)

        print(f"\nEpoch {epoch + 1} metrics:")
        print(f"Average training loss: {total_loss / len(train_dataloader):.4f}")
        print(f"Training accuracy: {correct / total:.4f}")

        print("\nValidation Results:")
        val_f1 = eval_model(model, val_dataloader, device)

        should_save = early_stopping(val_f1, epoch + 1)

        if should_save:
            best_val_f1 = val_f1
            save_model_checkpoint(model, optimizer, epoch, val_f1, checkpoint_dir, model.use_qlora)
            print(f"✅ Saved best model with val_f1: {val_f1:.4f}")

        if early_stopping.early_stop:
            print(f"🚨 Early stopping triggered at epoch {epoch + 1}")
            print(f"   - Best F1: {early_stopping.best_score:.4f}")
            print(f"   - Current F1: {val_f1:.4f}")
            print(f"   - Minimum epochs reached: {epoch + 1 >= min_epochs}")
            print(f"   - F1 requirement met: {val_f1 >= min_f1}")
            break

    if os.path.exists(checkpoint_dir):
        print("\n" + "="*80)
        print("Loading best model for testing...")
        print("="*80)
        load_model_checkpoint(model, None, checkpoint_dir, device)
        print("✅ Best model loaded successfully")
    else:
        print("⚠️ No checkpoint found, using final model for testing")

    return model


def run_test_inference(model, test_dataloader, device, dataset_id, seed, output_dir="./checkpoints"):
    """Run test inference and save predictions"""
    model.eval()
    predictions = []
    true_labels = []
    all_probabilities = []

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            c_idx = batch["c_idx"].to(device)
            s_idx = batch["s_idx"].to(device)
            r_idx = batch["r_idx"].to(device)
            e_idx = batch["e_idx"].to(device)

            outputs = model(input_ids, attention_mask, c_idx, s_idx, r_idx, e_idx)

            probabilities = torch.softmax(outputs["logits"], dim=1)
            if probabilities.dtype == torch.bfloat16:
                probabilities = probabilities.float()

            preds = torch.argmax(outputs["logits"], dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Force 3-class VERITE label space even if only 1–2 classes appear in y
    label_indices = list(range(len(LABEL_MAP)))      # [0, 1, 2]
    target_names = [INV_LABEL_MAP[i] for i in label_indices]

    report = classification_report(
        true_labels,
        predictions,
        labels=label_indices,       # <-- important change
        target_names=target_names,  # must match len(labels)
        digits=4,
        zero_division=0
    )

    print("\nDetailed Test Set Classification Report:")
    print(report)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)

    acc = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average="macro", zero_division=0)
    precision_macro = precision_score(true_labels, predictions, average="macro", zero_division=0)
    recall_macro = recall_score(true_labels, predictions, average="macro", zero_division=0)

    print(f"\nOverall Test Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")

    results = {
        'true_labels': true_labels,          # still ints 0/1/2
        'predictions': predictions,          # ints 0/1/2
        'probabilities': all_probabilities   # softmax over 3 VERITE classes
    }
    pred_filename = f"{output_dir}/test_results_{dataset_id}_seed{seed}.npy"
    np.save(pred_filename, results)
    print(f"\nTest results saved to '{pred_filename}'")

    return report


def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1, model_name="roberta-large"):
    """Setup LoRA for the model - prevents duplicate adapter issues"""
    if not hasattr(model.backbone, 'is_loaded_in_4bit') or not model.backbone.is_loaded_in_4bit:
        model.backbone = prepare_model_for_kbit_training(model.backbone)

    if hasattr(model.backbone, 'peft_config') and model.backbone.peft_config is not None:
        print("⚠️  Model already has LoRA adapters. Skipping new LoRA setup.")
        return model

    if "deberta" in model_name.lower():
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    elif "xlnet" in model_name.lower():
        target_modules = ["q", "k", "v", "o", "layer_1", "layer_2"]
    else:
        target_modules = ["query", "value", "key", "dense"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )

    if not isinstance(model.backbone, PeftModel):
        model.backbone = get_peft_model(model.backbone, lora_config)

    print("LoRA configuration:")
    print(f"  r: {lora_r}")
    print(f"  alpha: {lora_alpha}")
    print(f"  dropout: {lora_dropout}")
    print(f"  target_modules: {target_modules}")

    model.backbone.print_trainable_parameters()

    return model


def run_experiment_for_dataset_and_seed(dataset_id, dataset_config, seed, model_name="roberta-large",
                                        use_qlora=True, lora_r=16, lora_alpha=32, output_dir="./checkpoints"):
    """Run complete experiment for a dataset with a specific seed (Factify / VERITE etc.)"""
    print("\n" + "="*80)
    print(f"Running experiment for dataset '{dataset_id}' with seed {seed}")
    print(f"Model: {model_name} | QLoRA: {use_qlora}")
    print("="*80 + "\n")

    set_seed(seed)

    folder = dataset_config["folder"]
    train_path = f"{folder}/{dataset_config['train']}"
    val_path = f"{folder}/{dataset_config['val']}"
    test_path = f"{folder}/{dataset_config['test']}"

    config = MODEL_CONFIGS[model_name]
    max_len = config["max_position_embeddings"]

    print(f"Loading tokenizer and model ({model_name})...")
    print(f"Using max_length: {max_len} (native for this model)")

    tokenizer = config["tokenizer_class"].from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = FusedClassifier(
        model_name=model_name,
        num_labels=len(LABEL_MAP),
        special_tokens=SPECIAL_TOKENS,
        use_qlora=use_qlora
    )
    model.backbone.resize_token_embeddings(len(tokenizer))

    if use_qlora:
        model = setup_lora(model, lora_r=lora_r, lora_alpha=lora_alpha, model_name=model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not use_qlora:
        model.to(device)

    print("\nLoading datasets...")
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)

    print("\nDataset sizes:")
    print(f"Train: {len(train_data)}")
    print(f"Val:   {len(val_data)}")
    print(f"Test:  {len(test_data)}")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_data)
    print(f"Class weights: {class_weights}")

    train_dataset = FakeNewsDataset(train_data, tokenizer, max_len=max_len)
    val_dataset = FakeNewsDataset(val_data, tokenizer, max_len=max_len)
    test_dataset = FakeNewsDataset(test_data, tokenizer, max_len=max_len)

    batch_size = dataset_config.get("batch_size", 8)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=dataset_config.get("lr", 1e-4),
        eps=1e-8
    )

    epochs = dataset_config.get("epochs", 100)
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    model_short = model_name.split('/')[-1]
    checkpoint_dir = (
        f"{output_dir}/best_{dataset_id}_seed{seed}_{model_short}_qlora"
        if use_qlora
        else f"{output_dir}/best_{dataset_id}_seed{seed}_{model_short}"
    )

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")

    model = train_model(
        model, train_dataloader, val_dataloader, test_dataloader,
        optimizer, scheduler, device, class_weights=class_weights,
        epochs=epochs, patience=dataset_config.get("patience", 3),
        checkpoint_dir=checkpoint_dir,
        min_epochs=dataset_config.get("min_epochs", 11),
        min_f1=dataset_config.get("min_f1", 0.79)
    )

    print("\n" + "="*80)
    print("Running test inference...")
    print("="*80 + "\n")

    test_report = run_test_inference(model, test_dataloader, device, dataset_id, seed, output_dir)

    result_summary = {
        "dataset": dataset_id,
        "seed": seed,
        "model": model_name,
        "use_qlora": use_qlora,
        "max_length": max_len,
        "checkpoint_dir": checkpoint_dir,
        "test_report": test_report
    }

    summary_filename = f"{output_dir}/results_{dataset_id}_{model_short}_seed{seed}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=4)
    print(f"Result summary saved to '{summary_filename}'")

    return result_summary


def main():
    """Main function to run experiments on Factify / VERITE / similar JSON datasets"""
    parser = argparse.ArgumentParser(
        description='Train models with QLoRA on Factify / VERITE-style JSON datasets'
    )
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing the dataset files')
    parser.add_argument('--train_file', type=str,
                        default='train_evidence_qwen2vl_justification_final_justification.json',
                        help='Training data file (JSON)')
    parser.add_argument('--val_file', type=str,
                        default='val_evidence_qwen2vl_justification_final_justification.json',
                        help='Validation data file (JSON)')
    parser.add_argument('--test_file', type=str,
                        default='test_evidence_qwen2vl_justification_final_justification.json',
                        help='Test data file (JSON)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='Random seeds for experiments (e.g., --seeds 42 123 999)')
    parser.add_argument('--dataset_id', type=str, default='verite_text',
                        help='Dataset identifier for output files (e.g., verite_idefics3)')
    parser.add_argument('--model_name', type=str, default='roberta-large',
                        choices=['roberta-large', 'microsoft/deberta-v3-large', 'xlnet-large-cased'],
                        help='Model to use for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=4,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default 1e-4 for QLoRA)')
    parser.add_argument('--use_qlora', action='store_true', default=True,
                        help='Use QLoRA (4-bit quantization + LoRA)')
    parser.add_argument('--no_qlora', action='store_false', dest='use_qlora',
                        help='Disable QLoRA (use full model)')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory for saving models and results')
    parser.add_argument('--min_epochs', type=int, default=10,
                        help='Minimum epochs to run regardless of early stopping')
    parser.add_argument('--min_f1', type=float, default=0.65,
                        help='Minimum F1 score required before early stopping can trigger')
    parser.add_argument('--labels_csv', type=str, default=None,
                    help='Optional VERITE CSV file with gold labels (id + label columns)')


    args = parser.parse_args()

    # 🔹 Load VERITE labels from CSV into the global EXTERNAL_LABELS mapping
    global EXTERNAL_LABELS
    EXTERNAL_LABELS = {}
    if args.labels_csv is not None:
        # ✅ Use absolute path directly if given, otherwise join with data_dir
        if os.path.isabs(args.labels_csv):
            labels_path = args.labels_csv
        else:
            labels_path = os.path.join(args.data_dir, args.labels_csv)

        print(f"\nLoading VERITE labels from: {labels_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels CSV not found at: {labels_path}")
        EXTERNAL_LABELS = load_verite_labels(labels_path)
        print(f"Loaded {len(EXTERNAL_LABELS)} external labels.\n")


    global output_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_configs = {
        args.dataset_id: {
            "folder": args.data_dir,
            "train": args.train_file,
            "val": args.val_file,
            "test": args.test_file,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "min_epochs": args.min_epochs,
            "min_f1": args.min_f1
        }
    }

    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Max length: {MODEL_CONFIGS[args.model_name]['max_position_embeddings']} (native)")
    print(f"Use QLoRA: {args.use_qlora}")
    if args.use_qlora:
        print(f"LoRA rank (r): {args.lora_r}")
        print(f"LoRA alpha: {args.lora_alpha}")
    print(f"Data directory: {args.data_dir}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Test file: {args.test_file}")
    print(f"Seeds: {args.seeds}")
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Learning rate: {args.lr}")
    print("="*80 + "\n")

    overall_results = {}

    for dataset_id, dataset_config in dataset_configs.items():
        overall_results[dataset_id] = []

        for seed in args.seeds:
            result = run_experiment_for_dataset_and_seed(
                dataset_id,
                dataset_config,
                seed,
                model_name=args.model_name,
                use_qlora=args.use_qlora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                output_dir=args.output_dir
            )
            overall_results[dataset_id].append(result)

    model_short = args.model_name.split('/')[-1]
    overall_filename = f"{args.output_dir}/overall_results_{args.dataset_id}_{model_short}.json"
    with open(overall_filename, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=4)
    print("\n" + "="*80)
    print(f"Overall results saved to '{overall_filename}'")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
