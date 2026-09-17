import os
import json
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.optim import AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Mapping for LIAR-RAW labels (6 labels)
LIAR_RAW_LABELS = {
    "pants-fire": 0,
    "false": 1,
    "barely-true": 2,
    "half-true": 3,
    "mostly-true": 4,
    "true": 5
}
INV_LABEL_MAP = {v: k for k, v in LIAR_RAW_LABELS.items()}

class EarlyStopping:
    """Early stopping handler class"""
    def __init__(self, patience=3, min_delta=0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
        elif score <= (self.best_score + self.min_delta) if self.mode == 'max' else score >= (self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

def load_data(json_path):
    """Load dataset from JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

class FakeNewsDataset(Dataset):
    """Custom dataset for fake news classification with fixed max length (1024 tokens)"""
    def __init__(self, data, tokenizer, max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]
        support_justification = item.get("support_justification", "").strip() or "No support justification provided."
        refute_justification  = item.get("refute_justification", "").strip()  or "No refute justification provided."

        input_text = (
            f"Claim: {claim} [SEP] "
            f"Support Justification: {support_justification} [SEP] "
            f"Refute Justification: {refute_justification}"
        )

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len - 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids      = encoding["input_ids"].squeeze(0)[:self.max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)[:self.max_len]

        label_text = str(item.get("label", "")).lower()
        label = LIAR_RAW_LABELS.get(label_text, 1)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label":          torch.tensor(label, dtype=torch.long)
        }

def eval_model(model, dataloader, device, phase="Validation"):
    """Evaluate the model on a given dataloader"""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            truths.extend(labels.cpu().numpy())

    acc       = accuracy_score(truths, preds)
    f1        = f1_score(truths, preds, average="macro", zero_division=0)
    precision = precision_score(truths, preds, average="macro", zero_division=0)
    recall    = recall_score(truths, preds, average="macro", zero_division=0)

    print(f"{phase} Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")

    if phase == "Test":
        all_labels   = list(range(6))
        target_names = ["pants-fire","false","barely-true","half-true","mostly-true","true"]
        report = classification_report(
            truths, preds,
            labels=all_labels,
            target_names=target_names,
            digits=4,
            zero_division=0
        )
        print("\nDetailed Classification Report:")
        print(report)

    return f1

def train_model(
    model, train_loader, val_loader, test_loader,
    optimizer, scheduler, device,
    epochs=100, patience=3, checkpoint_filename="best_model.pth"
):
    """Train with early stopping; save best checkpoint and reload it."""
    early_stop = EarlyStopping(patience=patience)
    best_val_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        print(f"\nEpoch {epoch}/{epochs}")
        loop = tqdm(train_loader, leave=False)
        total_loss, correct, total = 0.0, 0, 0

        for batch in loop:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            loop.set_postfix(loss=loss.item(), accuracy=correct/total)

        print(f"→ Train loss: {total_loss/len(train_loader):.4f}  acc: {correct/total:.4f}")

        print("Validation Results:")
        val_f1 = eval_model(model, val_loader, device)

        early_stop(val_f1, model)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"New best val_f1 {val_f1:.4f}, saving to {checkpoint_filename}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1
            }, checkpoint_filename)

        if early_stop.early_stop:
            print(f" Early stopping at epoch {epoch}")
            break

    # Load best checkpoint if available
    if os.path.isfile(checkpoint_filename):
        print(f"Loading best checkpoint from '{checkpoint_filename}'")
        ckpt = torch.load(checkpoint_filename)  # no weights_only
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"No checkpoint found at '{checkpoint_filename}', using final model state")

    return model

def run_test_inference(model, test_loader, device, dataset_id):
    """Run test inference, save predictions, and print report."""
    model.eval()
    preds, truths, all_probs = [], [], []

    print("\nRunning inference on test dataset...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs   = torch.softmax(outputs.logits, dim=1)
            batch_preds = torch.argmax(outputs.logits, dim=1)

            preds.extend(batch_preds.cpu().numpy())
            truths.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels   = list(range(6))
    target_names = ["pants-fire","false","barely-true","half-true","mostly-true","true"]
    report = classification_report(
        truths, preds,
        labels=all_labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    print("\nDetailed Test Set Classification Report:")
    print(report)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(truths, preds, labels=all_labels)
    print("\nConfusion Matrix:")
    print(cm)

    results = {
        "true_labels": truths,
        "predictions": preds,
        "probabilities": all_probs
    }
    np.save(f"test_results_{dataset_id}.npy", results)
    print(f"\nTest results saved to 'test_results_{dataset_id}.npy'")

    return report

def run_experiment_for_dataset(dataset_id, cfg):
    print(f"\n***** Running experiment for '{dataset_id}' *****")
    folder     = cfg["folder"]
    train_path = f"{folder}/{cfg['train']}"
    val_path   = f"{folder}/{cfg['val']}"
    test_path  = f"{folder}/{cfg['test']}"

    max_len  = 1024
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    model     = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased", num_labels=6)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = load_data(train_path)
    val_data   = load_data(val_path)
    test_data  = load_data(test_path)

    train_ds = FakeNewsDataset(train_data, tokenizer, max_len)
    val_ds   = FakeNewsDataset(val_data,   tokenizer, max_len)
    test_ds  = FakeNewsDataset(test_data,  tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, sampler=SequentialSampler(val_ds))
    test_loader  = DataLoader(test_ds,  batch_size=8, sampler=SequentialSampler(test_ds))

    optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
    epochs    = 100
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    ckpt_name = f"best_{dataset_id}_xlnet.pth"
    print("Starting training...")
    model = train_model(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, device,
        epochs=epochs, patience=3, checkpoint_filename=ckpt_name
    )

    print("Running test inference...")
    test_report = run_test_inference(model, test_loader, device, dataset_id)

    summary = {
        "dataset":    dataset_id,
        "checkpoint": ckpt_name,
        "test_report": test_report
    }
    with open(f"results_{dataset_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Result summary saved to 'results_{dataset_id}.json'")

    return summary

def main():
    dataset_configs = {
        "mistral": {
            "folder": "cleaned_data/mistral",
            "train":  "generated_justifications_train_mistral.json",
            "val":    "generated_justifications_val_mistral.json",
            "test":   "generated_justifications_test_mistral.json"
        },
        "qwen": {
            "folder": "cleaned_data/qwen",
            "train":  "generated_justifications_train_qwen.json",
            "val":    "generated_justifications_val_qwen.json",
            "test":   "generated_justifications_test_qwen.json"
        },
        "gemma": {
            "folder": "cleaned_data/gemma",
            "train":  "generated_justifications_train_gemma.json",
            "val":    "generated_justifications_val_gemma.json",
            "test":   "generated_justifications_test_gemma.json"
        },
        "llama": {
            "folder": "cleaned_data/llama",
            "train":  "generated_justifications_train_llama.json",
            "val":    "generated_justifications_val_llama.json",
            "test":   "generated_justifications_test_llama.json"
        },
        "falcon": {
            "folder": "cleaned_data/falcon",
            "train":  "generated_justifications_train_falcon.json",
            "val":    "generated_justifications_val_falcon.json",
            "test":   "generated_justifications_test_falcon.json"
        },
    }

    import argparse, os
    ap = argparse.ArgumentParser(
        description="TBE-3 step 3: XLNet veracity prediction on LIAR-RAW.")
    ap.add_argument("--models", nargs="+", default=None, choices=list(dataset_configs),
                    help="Which GLM's justifications to train on. Default: all five.")
    ap.add_argument("--data_root", default=None,
                    help="Directory holding <model>/<split> files. Default: cleaned_data/<model>.")
    ap.add_argument("--output", default="overall_results.json")
    args = ap.parse_args()

    selected = {k: v for k, v in dataset_configs.items() if not args.models or k in args.models}
    if not selected:
        raise SystemExit(f"--models matched none of {sorted(dataset_configs)}")

    all_results = {}
    for ds_id, cfg in selected.items():
        cfg = dict(cfg)
        if args.data_root:
            cfg["folder"] = os.path.join(args.data_root, os.path.basename(cfg["folder"]))
        all_results[ds_id] = run_experiment_for_dataset(ds_id, cfg)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nOverall results saved to '{args.output}'")

if __name__ == "__main__":
    main()

