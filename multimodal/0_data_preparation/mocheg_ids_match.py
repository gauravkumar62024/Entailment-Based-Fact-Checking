import json

# === Input and output files ===
pred_file = "outputs/qwen2vl/val_predictions.json"            # Step 1 output (no paths yet)
paths_file = "outputs/val_image_paths_only.json"      # File with image paths


# Load JSONs
with open(pred_file, "r", encoding="utf-8") as f:
    predictions = json.load(f)

with open(paths_file, "r", encoding="utf-8") as f:
    img_data = json.load(f)

# Collect IDs
pred_ids = {str(item.get("Id", "")).strip() for item in predictions}
img_ids = {str(item.get("Id", "")).strip() for item in img_data}

# Find intersections and differences
matched_ids = sorted(pred_ids.intersection(img_ids))
missing_in_images = sorted(pred_ids - img_ids)
extra_in_images = sorted(img_ids - pred_ids)

# Report summary
print("\n=== ID Matching Summary ===")
print(f"🔗 Total IDs in predictions: {len(pred_ids)}")
print(f"🖼️ Total IDs with image paths: {len(img_ids)}")
print(f"✅ Matched IDs: {len(matched_ids)}")
print(f"⚠️ Missing in image paths: {len(missing_in_images)}")
print(f"ℹ️ Extra in image paths (not in predictions): {len(extra_in_images)}")

# Optional: preview some samples
print("\n--- Example Matched IDs ---")
print(matched_ids[:10])

print("\n--- Example Missing IDs (in predictions but no path) ---")
print(missing_in_images[:10])

print("\n--- Example Extra IDs (in image paths but not in predictions) ---")
print(extra_in_images[:10])

# Save matched IDs if you want to use them later
with open("matched_ids.json", "w", encoding="utf-8") as f:
    json.dump(matched_ids, f, indent=2)
print("\n💾 Saved matched IDs to matched_ids.json")
