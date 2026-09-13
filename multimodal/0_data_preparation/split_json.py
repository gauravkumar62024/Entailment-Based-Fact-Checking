import json, random

random.seed(42)

in_path  = "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_full_justifications_idefics3.json"
train_p  = "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_train.json"
val_p    = "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_val.json"
test_p   = "/data4/adityaks/multimodal_fact_checking/other_datasets/results/verite_justifications/idefics3_verite_full_test.json"

with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)

random.shuffle(data)

n = len(data)
n_train = int(0.8 * n)
n_val   = int(0.1 * n)
train = data[:n_train]
val   = data[n_train:n_train+n_val]
test  = data[n_train+n_val:]

for path, subset in [(train_p, train), (val_p, val), (test_p, test)]:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(path, len(subset))
