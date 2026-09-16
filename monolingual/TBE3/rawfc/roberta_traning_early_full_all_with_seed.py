import json
import torch
import copy
import numpy as np
import io
import contextlib
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
# Mapping for RAWFC labels
LABEL_MAP = {"false": 0, "half": 1, "true": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def set_seed(seed):
    
    # uncomment the following lines to set the seed for reproducibility
    
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    pass

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
        data = json.load(f)
    return data

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
        
        # Handle empty justifications
        true_justification = item.get("true_justification", "").strip()
        false_justification = item.get("false_justification", "").strip()
        if not true_justification:
            true_justification = "No true justification provided."
        if not false_justification:
            false_justification = "No false justification provided."
        # Concatenate claim and justifications
        input_text = f"Claim: {claim} [SEP] True Justification: {true_justification} [SEP] False Justification: {false_justification}"
        
        # Tokenization
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_len-2,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)[:self.max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)[:self.max_len]
    
        # Convert label to string to avoid type issues
        label_text = str(item.get("label", "")).lower()
        label = LABEL_MAP.get(label_text, 1)
    
        return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": torch.tensor(label, dtype=torch.long)
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
    """Train the model with early stopping; saves best model to the given checkpoint filename"""
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
    """Run test inference, save predictions, and print the detailed report.
       Predictions are saved in a file named with the dataset id and seed.
    """
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
    pred_filename = f"test_results_{dataset_id}_seed{seed}.npy"
    np.save(pred_filename, results)
    print(f"\nTest results saved to '{pred_filename}'")
    return report

def resize_position_embeddings(model, new_max_pos):
    current_max_pos, embed_size = model.roberta.embeddings.position_embeddings.weight.shape
    if new_max_pos > current_max_pos:
        # Create a new position embedding layer
        new_pos_embed = torch.nn.Embedding(new_max_pos, embed_size)

        # Copy existing embeddings
        new_pos_embed.weight.data[:current_max_pos, :] = model.roberta.embeddings.position_embeddings.weight.data

        # Copy the last embedding to new positions
        new_pos_embed.weight.data[current_max_pos:, :] = model.roberta.embeddings.position_embeddings.weight.data[-1, :].repeat(new_max_pos - current_max_pos, 1)

        # Assign new embeddings to model
        model.roberta.embeddings.position_embeddings = new_pos_embed
        model.config.max_position_embeddings = new_max_pos

        # Update token_type_ids buffer correctly
        model.roberta.embeddings.register_buffer(
            "token_type_ids",
            torch.zeros((1, new_max_pos), dtype=torch.long),
            persistent=False
        )
        print(f"Positional embeddings resized from {current_max_pos} to {new_max_pos}.")
    else:
        print("No resizing needed.")

    return model



def run_experiment_for_dataset_and_seed(dataset_id, dataset_config, seed):
    print(f"\n***** Running experiment for dataset '{dataset_id}' with seed {seed} *****")
    set_seed(seed)
    folder = dataset_config["folder"]
    train_path = f"{folder}/{dataset_config['train']}"
    val_path = f"{folder}/{dataset_config['val']}"
    test_path = f"{folder}/{dataset_config['test']}"
    max_len = 1024  # Using roberta-large's token size
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=3)
    model = resize_position_embeddings(model, 1024)
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
    epochs = 100  # Using default epochs; early stopping will likely trigger earlier.
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    checkpoint_filename = f"best_{dataset_id}_seed{seed}_roberta.pth"
    print("Starting training...")
    model = train_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, device, epochs=epochs, patience=3, checkpoint_filename=checkpoint_filename)
    print("Running test inference...")
    test_report = run_test_inference(model, test_dataloader, device, dataset_id, seed)
    result_summary = {
         "dataset": dataset_id,
         "seed": seed,
         "checkpoint": checkpoint_filename,
         "test_report": test_report
    }
    summary_filename = f"results_{dataset_id}_seed{seed}.json"
    with open(summary_filename, "w", encoding="utf-8") as f:
         json.dump(result_summary, f, indent=4)
    print(f"Result summary saved to '{summary_filename}'")
    return result_summary

def main():
    # Dataset configurations (as in roberta_traning_rawfc_for all.py)
    dataset_configs = {
         
         "mistral": {
              "folder": "mistral",
              "train": "train_mistral_cleaned.json",
              "val": "val_mistral_cleaned.json",
              "test": "test_mistral_cleaned.json"
         },
          "qwen": {
              "folder": "qwen2.5",
              "train": "train_qwen_cleaned.json",
              "val": "val_qwen_cleaned.json",
              "test": "test_qwen_cleaned.json"
         },
         "gemma": {
              "folder": "gemma",
              "train": "train_gemma_cleaned.json",
              "val": "val_gemma_cleaned.json",
              "test": "test_gemma_cleaned.json"
         },
         "llama": {
              "folder": "llama",
              "train": "train_meta_llama_cleaned.json",
              "val": "val_meta_llama_cleaned.json",
              "test": "test_meta_llama_cleaned.json"
         },
         "falcon": {
              "folder": "falcon",
              "train": "train_falcon_cleaned.json",
              "val": "val_falcon_cleaned.json",
              "test": "test_falcon_cleaned.json"
         },
        
         
    }
    import argparse, os
    ap = argparse.ArgumentParser(description="TBE-3 step 3: RoBERTa veracity prediction on RAW-FC.")
    ap.add_argument("--models", nargs="+", default=None, choices=list(dataset_configs),
                    help="Which GLM's justifications to train on. Default: all.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Random seeds to run (default: 42).")
    ap.add_argument("--data_root", default="cleaned_data",
                    help="Directory holding <folder>/<split> files (default: cleaned_data).")
    ap.add_argument("--output", default="overall_results.json")
    args = ap.parse_args()

    selected = {k: v for k, v in dataset_configs.items() if not args.models or k in args.models}
    if not selected:
        raise SystemExit(f"--models matched none of {sorted(dataset_configs)}")

    overall_results = {}
    for dataset_id, dataset_config in selected.items():
         cfg = dict(dataset_config)
         cfg["folder"] = os.path.join(args.data_root, cfg["folder"])
         overall_results[dataset_id] = []
         for seed in args.seeds:
             result = run_experiment_for_dataset_and_seed(dataset_id, cfg, seed)
             overall_results[dataset_id].append(result)
    overall_filename = args.output
    with open(overall_filename, "w", encoding="utf-8") as f:
         json.dump(overall_results, f, indent=4)
    print(f"\nOverall results saved to '{overall_filename}'")

if __name__ == "__main__":
    main()

