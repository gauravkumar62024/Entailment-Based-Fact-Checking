import os
import json
from sklearn.metrics import classification_report

# ------------------------------------
# LABEL MAPPINGS
# ------------------------------------
LABEL_MAP_3 = {
    "NEI": 0, 
    "not enough info": 0, 
    "Insufficient_Text": 0,
    "Insufficient_Multimodal": 0,
    "supported": 2,
    "Support_Multimodal": 2,
    "Support_Text": 2,
    "refuted": 1,
    "Refute": 1
}

def convert_to_3_label(label):
    """Map original labels to 3-class group."""
    return LABEL_MAP_3.get(label, None)

# ------------------------------------
# MAIN FUNCTION
# ------------------------------------
def generate_text_reports(base_dir="llm_prediction/factify", out_dir="factify_text_reports"):

    os.makedirs(out_dir, exist_ok=True)

    for dataset in os.listdir(base_dir):          # idifice, peligemma, qwenvl
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for model in os.listdir(dataset_path):     # llama, mistral
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path):
                continue

            for seed in os.listdir(model_path):    # seed42, seed57, seed196
                seed_path = os.path.join(model_path, seed)
                json_path = os.path.join(seed_path, "predictions_fixed.json")

                if not os.path.isfile(json_path):
                    continue

                print(f"Processing → {json_path}")

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                true_labels = [d["true_output"] for d in data]
                pred_labels = [d["predicted_output"] for d in data]

                # ----- 5-LABEL REPORT -----
                report_5 = classification_report(true_labels, pred_labels, digits=4)

                # ----- 3-LABEL CONVERSION -----
                true_3 = [convert_to_3_label(x) for x in true_labels]
                pred_3 = [convert_to_3_label(x) for x in pred_labels]

                # Remove invalid None labels
                valid_idx = [i for i in range(len(true_3)) if true_3[i] is not None and pred_3[i] is not None]
                true_3 = [true_3[i] for i in valid_idx]
                pred_3 = [pred_3[i] for i in valid_idx]

                report_3 = classification_report(true_3, pred_3, digits=4)

                # ----- SAVE TEXT FILE -----
                filename = f"{dataset}_{model}_{seed}.txt"
                filepath = os.path.join(out_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"=== FACTIFY CLASSIFICATION REPORT ===\n")
                    f.write(f"Dataset: {dataset}\nModel: {model}\nSeed: {seed}\n\n")

                    f.write("\n=====================\n 5-LABEL REPORT \n=====================\n")
                    f.write(report_5)

                    f.write("\n\n=====================\n 3-LABEL REPORT \n=====================\n")
                    f.write(report_3)

                print(f"Saved → {filepath}\n")

    print("\nAll text reports saved in:", out_dir)


# Run the report generator
if __name__ == "__main__":
    generate_text_reports()

