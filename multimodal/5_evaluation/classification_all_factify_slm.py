import os
import numpy as np
from sklearn.metrics import classification_report

# Root folder where everything is stored
root_dir = "slm_prediction"

# Mapping 5-class -> 3-class
five_to_three = {
    0: 2,   # Support_Text     -> Supported
    1: 2,   # Support_Multimodal -> Supported
    2: 0,   # Insufficient_Text -> NEI
    3: 0,   # Insufficient_Multimodal -> NEI
    4: 1    # Refute -> Refuted
}

target_names = ["NEI", "Refuted", "Supported"]

def process_npy_file(npy_path, report_filename):
    """
    Loads the .npy prediction file, converts labels, 
    computes 3-class report, and saves to a file.
    """
    try:
        data = np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Could not load: {npy_path}. Reason: {e}")
        return

    true_5 = data["true_labels"]
    pred_5 = data["predictions"]

    true_3 = [five_to_three[x] for x in true_5]
    pred_3 = [five_to_three[x] for x in pred_5]

    report = classification_report(true_3, pred_3, target_names=target_names, digits=4)

    # Save report
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"✅ Saved report to: {report_filename}")


# Walk through all nested folders
for dataset in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset)

    if not os.path.isdir(dataset_path):
        continue

    for subtask in os.listdir(dataset_path):
        subtask_path = os.path.join(dataset_path, subtask)

        if not os.path.isdir(subtask_path):
            continue

        for model in os.listdir(subtask_path):
            model_path = os.path.join(subtask_path, model)

            if not os.path.isdir(model_path):
                continue

            for seed in os.listdir(model_path):
                seed_path = os.path.join(model_path, seed)

                if not os.path.isdir(seed_path):
                    continue

                # Find .npy prediction file in the seed folder
                npy_files = [f for f in os.listdir(seed_path) if f.endswith(".npy")]

                if len(npy_files) == 0:
                    print(f"⚠️ No .npy file inside {seed_path}")
                    continue

                # Usually only 1 test_results file
                npy_file = npy_files[0]
                npy_path = os.path.join(seed_path, npy_file)

                # Construct output filename
                report_name = f"{dataset}_{subtask}_{model}_{seed}_report.txt"
                report_path = os.path.join(seed_path, report_name)

                # Process file
                process_npy_file(npy_path, report_path)

