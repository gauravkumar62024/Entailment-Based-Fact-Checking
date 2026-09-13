import os
import torch
import json
import argparse
import numpy as np
import neptune
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support
import neptune.new as neptune

# 1. Prompt construction
def construct_prompt(claim, tokenized_sents, max_sents=15):
    """
    Truncate context to max_sents for improved focus and smaller prompt.
    """
    
    truncated_sents = tokenized_sents[:max_sents]
    prompt = (
        "You are given a claim and some supporting context. "
        "Classify the claim as TRUE, FALSE, or HALF.Give answer in one word\n\n"
        f"Claim: {claim}\n\n"
        "Context:\n"
    )
    for i, sent in enumerate(truncated_sents, start=1):
        prompt += f"{i}) {sent}\n"
    prompt += "\nAnswer:"
    return prompt

# 2. Custom dataset
class VeracityDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_length=512):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                all_data = []

        self.samples = []
        for item in all_data:
            if all(key in item for key in ["claim", "label"]):
                # Ensure we have a list of tokenized sentences
                item['tokenized'] = item.get('tokenized', [])
                self.samples.append(item)
            else:
                print(f"Skipping item due to missing keys: {item.get('claim', 'No claim')}")

        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loaded {len(self.samples)} valid samples from {jsonl_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        claim = item["claim"]
        tokenized_sents = item.get("tokenized", [])
        label = item["label"].strip()

        # Build prompt
        prompt = construct_prompt(claim, tokenized_sents, max_sents=5)
        full_text = prompt + " " + label  # e.g. "... Answer: TRUE"

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        prompt_len = len(self.tokenizer(prompt).input_ids)

        # Mask out prompt tokens in the labels
        labels = tokenized["input_ids"].copy()
        for i in range(prompt_len):
            if i < self.max_length:
                labels[i] = -100
        tokenized["labels"] = labels

        # Convert to tensors
        return {key: torch.tensor(val) for key, val in tokenized.items()}

# 3. Optional: custom metric function
#    Note: If you feed the entire LM vocabulary, this can be huge. We will do a simpler approach below.
def compute_metrics(pred):
    # pred.predictions: shape [batch_size, seq_len, vocab_size]
    # This can blow up memory for large LLMs. 
    # We'll keep it here, but we won't pass it to the Trainer to avoid spikes.
    if isinstance(pred.predictions, tuple):
        preds = pred.predictions[0]
    else:
        preds = pred.predictions

    # Argmax over vocab dimension
    preds = torch.from_numpy(preds) if isinstance(preds, np.ndarray) else preds
    preds = torch.argmax(preds, dim=-1)

    labels = torch.from_numpy(pred.label_ids) if isinstance(pred.label_ids, np.ndarray) else pred.label_ids
    mask = labels != -100
    preds = preds[mask]
    labels = labels[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu().numpy(),
        preds.cpu().numpy(),
        average='weighted'
    )
    return {'precision': precision, 'recall': recall, 'f1': f1}

# 4. Custom evaluation in small chunks to avoid big GPU usage
def evaluate_in_chunks(trainer, val_dataloader, device="cuda"):
    """
    Runs a forward pass in small batches, collecting predictions and labels,
    then calculates weighted precision, recall, f1.
    """
    trainer.model.eval()
    all_preds, all_labels = [], []

    for batch in val_dataloader:
        # Move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = trainer.model(**batch)
            # outputs.logits: shape [batch_size, seq_len, vocab_size]
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

            labels = batch["labels"]
            mask = labels != -100
            preds = preds[mask]
            labels = labels[mask]

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    return {"precision": precision, "recall": recall, "f1": f1}

# 5. Main function
def main(args):
    run = neptune.init_run(
        project="gaurav22/llama31loratraning",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1M2RhODg5MC0wZDU3LTQxMzUtOWIwZC0wMmNiMmViNmM4ZmQifQ==",
    )
    run["parameters"] = vars(args)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_name = args.model_name or "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )

    # Prepare for k-bit training + LoRA
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query_key_value"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = VeracityDataset(args.train_data, tokenizer)
    val_dataset = VeracityDataset(args.val_data, tokenizer)

    # We'll create our own val_dataloader for custom evaluation
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,  # Keep it small to avoid big GPU usage
        shuffle=False
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    output_dir = args.output_dir or "./lora-veracity-checkpoint_falcon"
    os.makedirs(output_dir, exist_ok=True)

    # IMPORTANT: Disable built-in frequent eval to prevent memory blowups
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # also small
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # Turn off or reduce the built-in evaluation
        evaluation_strategy="no",    # <---- no built-in evaluation
        # We won't use these now, but you can keep them for manual calls
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"  # or "neptune" if you want direct logging
    )

    # We do NOT set eval_dataset here, to avoid storing massive logits on GPU
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics  # omit or set to None
    )

    # Train
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Now do custom evaluation on the entire validation set in small chunks
    eval_results = evaluate_in_chunks(trainer, val_dataloader, device=trainer.args.device)
    run["eval_results"] = eval_results
    print("Final Evaluation Results:", eval_results)

    run.stop()

# 6. Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune Llama-3.1 with LoRA for Veracity Prediction")
    parser.add_argument("--model_name", default=None, help="Pretrained model name")
    parser.add_argument("--train_data", required=True, help="Path to training data")
    parser.add_argument("--val_data", required=True, help="Path to validation data")
    parser.add_argument("--output_dir", default=None, help="Output directory for checkpoints")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

    
    

