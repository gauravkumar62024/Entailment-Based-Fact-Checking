import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from transformers import BitsAndBytesConfig
from peft import PeftModel

# ===== IMPORT FROM TRAINING FILE =====
# make sure inference.py is in same folder or PYTHONPATH
from train_xfact_entialment import (
    MODEL_CONFIGS, LABEL_MAP, INV_LABEL_MAP,
    SPECIAL_TOKENS, FakeNewsDataset,
    FusedClassifier, load_model_checkpoint
)

# =====================================================
def load_trained_model(
    checkpoint_dir,
    model_name,
    num_labels=7,
    use_qlora=True,
    device="cuda"
):
    model = FusedClassifier(
        model_name=model_name,
        num_labels=num_labels,
        special_tokens=SPECIAL_TOKENS,
        use_qlora=use_qlora
    )
    tokenizer = MODEL_CONFIGS[model_name]["tokenizer_class"].from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = FusedClassifier(
        model_name=model_name,
        num_labels=7,
        special_tokens=SPECIAL_TOKENS,
        use_qlora=True
    )

    # 🔥 CRITICAL LINE (MISSING)
    model.backbone.resize_token_embeddings(len(tokenizer))

    # NOW load LoRA + classifier
    load_model_checkpoint(model, None, checkpoint_dir, device)
        
    # load LoRA + classifier head
    # load_model_checkpoint(model, optimizer=None,
    #                       checkpoint_dir=checkpoint_dir,
    #                       device=device)

    model.eval()
    return model


# =====================================================
def run_test_inference(
    model,
    test_data,
    tokenizer,
    batch_size,
    device,
    output_dir,
    dataset_id,
    seed
):
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = FakeNewsDataset(
        test_data,
        tokenizer,
        max_len=MODEL_CONFIGS[model.model_name]["max_position_embeddings"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(test_dataset)
    )

    preds, golds, probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids,
                attention_mask,
                batch["c_idx"].to(device),
                batch["s_idx"].to(device),
                batch["r_idx"].to(device),
                batch["e_idx"].to(device),
            )

            logits = outputs["logits"]
            prob = torch.softmax(logits, dim=1)

            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            golds.extend(labels.cpu().numpy())
            probs.extend(prob.float().cpu().numpy())

    # ===== Metrics =====
    acc = accuracy_score(golds, preds)
    f1 = f1_score(golds, preds, average="macro")
    prec = precision_score(golds, preds, average="macro", zero_division=0)
    rec = recall_score(golds, preds, average="macro", zero_division=0)
    cm = confusion_matrix(golds, preds)

    report = classification_report(
        golds, preds,
        target_names=list(LABEL_MAP.keys()),
        digits=4,
        zero_division=0
    )

    # ===== Save =====
    np.save(
        f"{output_dir}/test_predictions_{dataset_id}_seed{seed}.npy",
        {
            "gold": golds,
            "pred": preds,
            "probabilities": probs
        }
    )

    # human-readable CSV
# ===== Save predictions as JSON =====
    output_json = []

    for idx, (g, p, pr) in enumerate(zip(golds, preds, probs)):
        output_json.append({
            "claim": test_data[idx].get("claim", ""),
            "true_label": INV_LABEL_MAP[int(g)],
            "predicted_label": INV_LABEL_MAP[int(p)],
            "probabilities": pr.tolist()
        })


    json_path = f"{output_dir}/test_predictions_{dataset_id}_seed{seed}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    print(f"✅ Predictions saved to {json_path}")


    # metrics txt
    with open(f"{output_dir}/metrics_{dataset_id}_seed{seed}.txt", "w") as f:
        f.write(f"""
            Accuracy: {acc:.4f}
            Macro-F1: {f1:.4f}
            Macro-Precision: {prec:.4f}
            Macro-Recall: {rec:.4f}

            Classification Report:
            {report}

            Confusion Matrix:
            {cm}
            """)

    print("✅ Inference complete")
    print(report)


# =====================================================
if __name__ == "__main__":
    # -------- CONFIG --------
    checkpoint_dir = "./checkpoints/best_xfact_qwen_seed42_xlm-roberta-large_qlora"
    test_json = "../Entailment_based_factchecking/output_of_justification_generation/xfact/qwen/Indomain_semantic_chunking_justifications.json"
    model_name = "xlm-roberta-large"
    dataset_id = "xfact_qwen"
    seed = 42
    batch_size = 8
    output_dir = "./checkpoints_new"
    use_qlora = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- LOAD --------
    with open(test_json, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    tokenizer = MODEL_CONFIGS[model_name]["tokenizer_class"].from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = load_trained_model(
        checkpoint_dir,
        model_name,
        use_qlora=use_qlora,
        device=device
    )

    run_test_inference(
        model,
        test_data,
        tokenizer,
        batch_size,
        device,
        output_dir,
        dataset_id,
        seed
    )
