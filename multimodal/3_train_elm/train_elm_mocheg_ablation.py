import json
import torch
import numpy as np
import random
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from transformers import (
    DebertaV2Tokenizer, DebertaV2Model,
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

# Mapping for 3-way classification (supported, NEI, refuted)
LABEL_MAP = {
    "supported": 0,
    "NEI": 1,
    "refuted": 2
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, mode='max', min_epochs=10, min_f1=0.65):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.min_f1 = min_f1
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.current_epoch = 0

    def __call__(self, score, current_epoch):
        self.current_epoch = current_epoch
        
        if current_epoch < self.min_epochs and score < self.min_f1:
            print(f"⚠️ Early stopping suspended: epoch {current_epoch} < {self.min_epochs} and F1 {score:.4f} < {self.min_f1}")
            return True
        
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'max':
            improvement = score > (self.best_score + self.min_delta)
        else:
            improvement = score < (self.best_score - self.min_delta)
            
        if improvement:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience and (current_epoch >= self.min_epochs or score >= self.min_f1):
                self.early_stop = True
                print(f"🚨 Early stopping triggered: {self.counter} epochs without improvement")
            return False

def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data file {json_path} not found")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples from {json_path}")
    return data

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_input_string(claim: str, support: str = None, refute: str = None, ablation_type: str = "full") -> tuple:
    """
    Build input string based on ablation type
    ablation_type: 'full', 'claim_support', 'claim_refute', 'claim_only'
    Returns: (input_text, special_tokens_list)
    """
    if ablation_type == "full":
        special_tokens = ["<CLAIM>", "<SUPPORT>", "<REFUTE>", "<END>"]
        input_text = f"<CLAIM> {claim} <SUPPORT> {support} <REFUTE> {refute} <END>"
    elif ablation_type == "claim_support":
        special_tokens = ["<CLAIM>", "<SUPPORT>", "<END>"]
        input_text = f"<CLAIM> {claim} <SUPPORT> {support} <END>"
    elif ablation_type == "claim_refute":
        special_tokens = ["<CLAIM>", "<REFUTE>", "<END>"]
        input_text = f"<CLAIM> {claim} <REFUTE> {refute} <END>"
    elif ablation_type == "claim_only":
        special_tokens = ["<CLAIM>", "<END>"]
        input_text = f"<CLAIM> {claim} <END>"
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")
    
    return input_text, special_tokens

class MochegAblationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, ablation_type="full"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ablation_type = ablation_type

    def __len__(self):
        return len(self.data)

    def _truncate_smart(self, input_ids, attention_mask, special_token_ids):
        actual_len = len(input_ids)
        if actual_len <= self.max_len:
            pad_len = self.max_len - actual_len
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
            return input_ids, attention_mask
        
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
        
        claim = clean_text(item.get("claim_text", item.get("claim", "")))
        supporting_justification = clean_text(item.get("supporting_text_evidence", item.get("support_justification", "")))
        refuting_justification = clean_text(item.get("refuting_text_evidence", item.get("refute_justification", "")))
        
        input_text, special_tokens = build_input_string(
            claim, supporting_justification, refuting_justification, self.ablation_type
        )
        
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
        special_token_ids = [tid(token) for token in special_tokens]
        
        input_ids, attention_mask = self._truncate_smart(input_ids, attention_mask, special_token_ids)
        
        actual_len = int(attention_mask.sum().item())
        actual_len = max(5, min(actual_len, self.max_len))
        
        # Find special token positions based on ablation type
        token_positions = {}
        for token in special_tokens:
            t_id = tid(token)
            pos = (input_ids == t_id).nonzero(as_tuple=False)
            token_positions[token] = int(pos[0].item()) if len(pos) > 0 else 0
        
        # Set positions based on what tokens exist
        c_idx = token_positions.get("<CLAIM>", 1)
        s_idx = token_positions.get("<SUPPORT>", min(c_idx + 2, actual_len - 2))
        r_idx = token_positions.get("<REFUTE>", min(s_idx + 2, actual_len - 1))
        e_idx = token_positions.get("<END>", actual_len - 1)
        
        # Ensure valid ordering
        c_idx = max(0, min(c_idx, actual_len - 4))
        s_idx = max(c_idx + 1, min(s_idx, actual_len - 3))
        r_idx = max(s_idx + 1, min(r_idx, actual_len - 2))
        e_idx = max(r_idx + 1, min(e_idx, actual_len - 1))
        
        label_text = str(item.get("label", "refuted"))
        label = LABEL_MAP.get(label_text, LABEL_MAP["refuted"])
    
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "c_idx": torch.tensor(c_idx, dtype=torch.long),
            "s_idx": torch.tensor(s_idx, dtype=torch.long),
            "r_idx": torch.tensor(r_idx, dtype=torch.long),
            "e_idx": torch.tensor(e_idx, dtype=torch.long)
        }

class FusedClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, special_tokens: list, use_qlora: bool = True):
        super().__init__()
        self.model_name = model_name
        self.use_qlora = use_qlora

        if use_qlora:
            print(f"Loading {model_name} with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.backbone = DebertaV2Model.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            device = next(self.backbone.parameters()).device
            dtype = torch.bfloat16
        else:
            self.backbone = DebertaV2Model.from_pretrained(model_name)
            device = torch.device("cpu")
            dtype = torch.float32

        hidden = 1024
        
        self.classifier = nn.Linear(hidden, num_labels, dtype=dtype, device=device)
        self.proj = nn.Linear(hidden, hidden, dtype=dtype, device=device)
        
        self.special_tokens = special_tokens

    def forward(self, input_ids, attention_mask, c_idx, s_idx, r_idx, e_idx, labels=None):
        B, L = input_ids.shape
        
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = outputs.last_hidden_state
        
        attention_mask = attention_mask.to(last.dtype)
        
        cls_pool = (last * attention_mask.unsqueeze(-1)).sum(1) / (attention_mask.sum(1, keepdim=True) + 1e-9)
        logits = self.classifier(cls_pool)
        
        ce_loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(logits, labels)
        
        return {"logits": logits, "loss": ce_loss, "ce": ce_loss}

def eval_model(model, dataloader, device, phase="Validation"):
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
        target_names = [INV_LABEL_MAP[i] for i in range(len(LABEL_MAP))]
        report = classification_report(true_labels, predictions, 
                                         target_names=target_names, 
                                         digits=4,
                                         zero_division=0)
        print("\nDetailed Classification Report:")
        print(report)
    
    return f1, report if phase == "Test" else None

def compute_class_weights(train_data):
    labels = [LABEL_MAP.get(str(item.get("label", "refuted")), LABEL_MAP["refuted"]) for item in train_data]
    classes = np.arange(len(LABEL_MAP))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float)

def save_model_checkpoint(model, optimizer, epoch, val_f1, checkpoint_dir, use_qlora):
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
                print(f"⚠️ Could not load LoRA adapters: {e}")
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
                min_epochs=10, min_f1=0.65):
    
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
        val_f1, _ = eval_model(model, val_dataloader, device)
        
        should_save = early_stopping(val_f1, epoch + 1)
        
        if should_save:
            best_val_f1 = val_f1
            save_model_checkpoint(model, optimizer, epoch, val_f1, checkpoint_dir, model.use_qlora)
            print(f"✅ Saved best model with val_f1: {val_f1:.4f}")
        
        if early_stopping.early_stop:
            print(f"🚨 Early stopping triggered at epoch {epoch + 1}")
            break
    
    if os.path.exists(checkpoint_dir):
        print(f"\n{'='*80}")
        print("Loading best model for testing...")
        print(f"{'='*80}")
        load_model_checkpoint(model, None, checkpoint_dir, device)
        print(f"✅ Best model loaded successfully")
    
    return model

def run_test_inference(model, test_dataloader, device, output_path):
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
    
    target_names = [INV_LABEL_MAP[i] for i in range(len(LABEL_MAP))]
    report = classification_report(true_labels, predictions, target_names=target_names, digits=4, zero_division=0)
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
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': all_probabilities,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'metrics': {
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro)
        }
    }
    
    np.save(output_path.replace('.json', '.npy'), {
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': all_probabilities
    })
    
    with open(output_path, 'w') as f:
        json.dump({
            'report': report,
            'confusion_matrix': cm.tolist(),
            'metrics': results['metrics']
        }, f, indent=4)
    
    print(f"\nTest results saved to '{output_path}'")
    
    return report, results['metrics']

def setup_lora(model, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    
    if not hasattr(model.backbone, 'is_loaded_in_4bit') or not model.backbone.is_loaded_in_4bit:
        model.backbone = prepare_model_for_kbit_training(model.backbone)
    
    if hasattr(model.backbone, 'peft_config') and model.backbone.peft_config is not None:
        print("⚠️ Model already has LoRA adapters. Skipping new LoRA setup.")
        return model
    
    target_modules = ["query_proj", "key_proj", "value_proj", "dense"]
    
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

def run_ablation_experiment(ablation_type, seed, args):
    """
    Run single ablation experiment
    ablation_type: 'claim_support', 'claim_refute', 'claim_only'
    """
    print(f"\n{'='*80}")
    print(f"ABLATION: {ablation_type.upper()} | SEED: {seed}")
    print(f"{'='*80}\n")
    
    set_seed(seed)
    
    print(f"Loading tokenizer and model (microsoft/deberta-v3-large)...")
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-large")
    
    # Add special tokens based on ablation type
    _, special_tokens = build_input_string("", "", "", ablation_type)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    model = FusedClassifier(
        model_name="microsoft/deberta-v3-large", 
        num_labels=3, 
        special_tokens=special_tokens, 
        use_qlora=args.use_qlora
    )
    model.backbone.resize_token_embeddings(len(tokenizer))
    
    if args.use_qlora:
        model = setup_lora(model, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not args.use_qlora:
        model.to(device)
    else:
        if hasattr(model.backbone, 'device'):
            device = model.backbone.device
        else:
            device = next(model.backbone.parameters()).device
        print(f"Model is on device: {device}")
    
    print(f"\nLoading datasets...")
    train_path = f"{args.data_dir}/{args.train_file}"
    val_path = f"{args.data_dir}/{args.val_file}"
    test_path = f"{args.data_dir}/{args.test_file}"
    
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    
    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_data)
    print(f"Class weights: {class_weights}")
    
    train_dataset = MochegAblationDataset(train_data, tokenizer, max_len=512, ablation_type=ablation_type)
    val_dataset = MochegAblationDataset(val_data, tokenizer, max_len=512, ablation_type=ablation_type)
    test_dataset = MochegAblationDataset(test_data, tokenizer, max_len=512, ablation_type=ablation_type)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=RandomSampler(train_dataset))
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=SequentialSampler(val_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=SequentialSampler(test_dataset))
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=args.lr, eps=1e-8)
    
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    
    checkpoint_dir = f"{args.output_dir}/ablation_{ablation_type}_seed{seed}"
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    model = train_model(model, train_dataloader, val_dataloader, test_dataloader, 
                       optimizer, scheduler, device, class_weights=class_weights,
                       epochs=args.epochs, patience=args.patience, 
                       checkpoint_dir=checkpoint_dir,
                       min_epochs=args.min_epochs,
                       min_f1=args.min_f1)
    
    print("\n" + "="*80)
    print("Running test inference...")
    print("="*80 + "\n")
    
    output_path = f"{args.output_dir}/results_ablation_{ablation_type}_seed{seed}.json"
    test_report, metrics = run_test_inference(model, test_dataloader, device, output_path)
    
    result_summary = {
        "ablation_type": ablation_type,
        "seed": seed,
        "model": "microsoft/deberta-v3-large",
        "use_qlora": args.use_qlora,
        "checkpoint_dir": checkpoint_dir,
        "test_report": test_report,
        "metrics": metrics
    }
    
    return result_summary

def main():
    parser = argparse.ArgumentParser(description='Ablation Study on Mocheg Dataset with DeBERTa')
    parser.add_argument('--data_dir', type=str, 
                        default='mocheg_justification/qwen2vl',
                        help='Data directory')
    parser.add_argument('--train_file', type=str, 
                        default='train_justifications_final.json',
                        help='Training data file')
    parser.add_argument('--val_file', type=str,
                        default='val_justifications_final.json',
                        help='Validation data file')
    parser.add_argument('--test_file', type=str,
                        default='test_justifications_final.json',
                        help='Test data file')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 57, 196],
                        help='Random seeds for experiments')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=4,
                        help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--use_qlora', action='store_true', default=True,
                        help='Use QLoRA (4-bit quantization + LoRA)')
    parser.add_argument('--no_qlora', action='store_false', dest='use_qlora',
                        help='Disable QLoRA')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha')
    parser.add_argument('--output_dir', type=str, 
                        default='./mocheg_ablation_results',
                        help='Output directory for saving models and results')
    parser.add_argument('--min_epochs', type=int, default=10,
                        help='Minimum epochs to run')
    parser.add_argument('--min_f1', type=float, default=0.65,
                        help='Minimum F1 score for early stopping')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ABLATION STUDY CONFIGURATION")
    print("="*80)
    print(f"Model: microsoft/deberta-v3-large")
    print(f"Dataset: Mocheg (Qwen2VL)")
    print(f"Use QLoRA: {args.use_qlora}")
    print(f"Seeds: {args.seeds}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.epochs}")
    print(f"Patience: {args.patience}")
    print(f"Learning rate: {args.lr}")
    print(f"Data directory: {args.data_dir}")
    print("="*80 + "\n")
    
    ablation_types = ['claim_support', 'claim_refute', 'claim_only']
    
    all_results = {}
    
    for ablation_type in ablation_types:
        all_results[ablation_type] = []
        
        for seed in args.seeds:
            print(f"\n{'#'*80}")
            print(f"# Running: {ablation_type.upper()} with seed {seed}")
            print(f"{'#'*80}\n")
            
            result = run_ablation_experiment(ablation_type, seed, args)
            all_results[ablation_type].append(result)
    
    summary_file = f"{args.output_dir}/ablation_study_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"ABLATION STUDY COMPLETED")
    print(f"{'='*80}")
    print(f"All results saved to: {summary_file}")
    print(f"\nSummary of experiments:")
    for ablation_type in ablation_types:
        print(f"\n{ablation_type.upper()}:")
        for result in all_results[ablation_type]:
            print(f"  Seed {result['seed']}: F1={result['metrics']['f1_macro']:.4f}, Acc={result['metrics']['accuracy']:.4f}")

if __name__ == "__main__":
    main()
