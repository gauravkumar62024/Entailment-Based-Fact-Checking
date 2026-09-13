import json
from collections import Counter

def count_unique_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [str(item.get("label", "")).strip() for item in data]
    counter = Counter(labels)

    print("Unique labels and counts:\n")
    for label, count in counter.items():
        print(f"{label}: {count}")

    print("\nTotal unique labels:", len(counter))

# Example usage:
count_unique_labels("val_justifications_final.json")

