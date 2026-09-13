import json
import torch
import copy
import numpy as np
import io
import contextlib
import random
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import XLNetTokenizer, XLNetForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Mapping for labels (modify if necessary)
LABEL_MAP = {"false": 0, "half": 1, "true": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """Early stopping handler class."""
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
    """Load dataset from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

class FakeNewsDataset(Dataset):
    """
    Custom dataset for claim and understanding classification with fixed max length.
    Expects each JSON entry to have keys "claim" and "understanding".
    """
    def __init__(self, data, tokenizer, max_len=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        claim = item["claim"]

        # Use the "understanding" field; provide a default message if missing.
        understanding = item.get("understanding", "").strip()
        if not understanding:
            understanding = "No understanding provided."

        # Build the input text as "Claim: ... [SEP] Understanding: ..."
        input_text = f"Claim: {claim} [SEP] Understanding: {understanding}"

        # Tokenize the input
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len - 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)[:self.max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)[:self.max_len]

        # Process label (if available)
        label_text = str(item.get("label", "")).lower()
        label = LABEL_MAP.get(label_text, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }

def eval_model(model, dataloader, device, phase="Validation"):
    """Evaluate the model on a given dataloader."""
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="macro")
    precision = precision_score(true_labels, predictions, average="macro")
    recall = recall_score(true_labels, predictions, average="macro")
    print(f"{phase} Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if phase == "Test":
        report = classification_report(true_labels, predictions, 
                                         target_names=["False", "Half-True", "True"], 
                                         digits=4)
        print("\nDetailed Classification Report:")
        print(report)
    return f1

def train_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, device, epochs=100, patience=3, checkpoint_filename="best_model.pth"):
    """Train the model with early stopping and save the best model."""
    early_stopping = EarlyStopping(patience=patience)
    best_val_f1 = 0
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
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix(loss=loss.item(), accuracy=correct / total)
        print(f"\nEpoch {epoch + 1} metrics:")
        print(f"Average training loss: {total_loss / len(train_dataloader):.4f}")
        print(f"Training accuracy: {correct / total:.4f}")
        print("\nValidation Results:")
        val_f1 = eval_model(model, val_dataloader, device)
        early_stopping(val_f1, model)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
            }, checkpoint_filename)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_test_inference(model, test_dataloader, device, dataset_id, seed):
    """Run test inference, save predictions, and print a detailed report."""
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
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    report = classification_report(true_labels, predictions, target_names=["False", "Half-True", "True"], digits=4)
    print("\nDetailed Test Set Classification Report:")
    print(report)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predictions)
    print("\nConfusion Matrix:")
    print(cm)
    results = {
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': all_probabilities
    }
    pred_filename = f"xlnet_large_results/test_results_{dataset_id}_seed{seed}.npy"
    np.save(pred_filename, results)
    print(f"\nTest results saved to '{pred_filename}'")
    return report

def run_experiment_for_dataset_and_seed(dataset_id, dataset_config, seed):
    print(f"\n***** Running experiment for dataset '{dataset_id}' with seed {seed} *****")
    set_seed(seed)
    folder = dataset_config["folder"]
    train_path = f"{folder}/{dataset_config['train']}"
    val_path = f"{folder}/{dataset_config['val']}"
    test_path = f"{folder}/{dataset_config['test']}"
    max_len = 1024  # Maximum token length for XLNet
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    model = XLNetForSequenceClassification.from_pretrained("xlnet-large-cased", num_labels=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    train_dataset = FakeNewsDataset(train_data, tokenizer, max_len=max_len)
    val_dataset = FakeNewsDataset(val_data, tokenizer, max_len=max_len)
    test_dataset = FakeNewsDataset(test_data, tokenizer, max_len=max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=8, sampler=RandomSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=8, sampler=SequentialSampler(val_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=8, sampler=SequentialSampler(test_dataset))
    optimizer = AdamW(model.parameters(), lr=1e-05, eps=1e-8)
    epochs = 100  # Set as needed; early stopping will likely halt before 100 epochs.
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    # Create the output folder if it does not exist
    os.makedirs("xlnet_large_results", exist_ok=True)
    
    checkpoint_filename = f"xlnet_large_results/best_{dataset_id}_seed{seed}_xlnet.pth"
    print("Starting training...")
    model = train_model(model, train_dataloader, val_dataloader, test_dataloader,
                        optimizer, scheduler, device, epochs=epochs, patience=3,
                        checkpoint_filename=checkpoint_filename)
    print("Running test inference...")
    test_report = run_test_inference(model, test_dataloader, device, dataset_id, seed)
    result_summary = {
         "dataset": dataset_id,
         "seed": seed,
         "checkpoint": checkpoint_filename,
         "test_report": test_report
    }
    summary_filename = f"xlnet_large_results/results_{dataset_id}_seed{seed}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
         json.dump(result_summary, f, indent=4)
    print(f"Result summary saved to '{summary_filename}'")
    return result_summary

def main():
    # Dataset configurations for the claim and understanding task.
    # Adjust the file names to your understanding-specific JSON files.
    dataset_configs = {
         "mistral": {
              "folder": "mistral",
              "train": "understanding_train_mistral.json",
              "val": "understanding_val_mistral.json",
              "test": "understanding_test_mistral.json"
         },
         "llama": {
              "folder": "llama",
              "train": "understanding_train_llama.json",
              "val": "understanding_val_llama.json",
              "test": "understanding_test_llama.json"
         },
         "qwen": {
              "folder": "qwen2.5",
              "train": "understanding_train_qwen.json",
              "val": "understanding_val_qwen.json",
              "test": "understanding_test_qwen.json"
         },
         "gemma": {
              "folder": "gemma",
              "train": "understanding_train_gemma.json",
              "val": "understanding_val_gemma.json",
              "test": "understanding_test_gemma.json"
         },
         
         "falcon": {
              "folder": "falcon",
              "train": "understanding_train_falcon.json",
              "val": "understanding_val_falcon.json",
              "test": "understanding_test_falcon.json"
         },
    }
    seeds = [42, 123, 999]
    overall_results = {}
    for dataset_id, dataset_config in dataset_configs.items():
         overall_results[dataset_id] = []
         for seed in seeds:
              result = run_experiment_for_dataset_and_seed(dataset_id, dataset_config, seed)
              overall_results[dataset_id].append(result)
    overall_filename = "xlnet_large_results/overall_results.json"
    with open(overall_filename, "w", encoding="utf-8") as f:
         json.dump(overall_results, f, indent=4)
    print(f"\nOverall results saved to '{overall_filename}'")

if __name__ == "__main__":
    main()

