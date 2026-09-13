import json
import os
new_instruction = """Classify the given claim into one of the six categories (TRUE, MOSTLY-TRUE, HALF-TRUE, BARELY-TRUE, FALSE, PANTS-FIRE) based on the overall understanding of the provided evidence:\n\n1. TRUE: Fully supported by the context.\n2. MOSTLY-TRUE: Mostly supported but with some minor inaccuracies.\n3. HALF-TRUE: Partially supported, with significant omissions or misleading aspects.\n4. BARELY-TRUE: Contains some element of truth but is mostly misleading or inaccurate.\n5. FALSE: Contradicted or unsupported by the context.\n6. PANTS-FIRE: Completely false and outrageously ridiculous.\n\nProvide exactly one label."""



root_dir = "lairraw"  # <-- Change this to your actual root folder path

# Walk through all subdirectories in the root directory.
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        # Process only JSON files.
        if filename.endswith('.json'):
            input_file_path = os.path.join(subdir, filename)
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f" Could not load {input_file_path}: {e}")
                continue

            transformed_data = []
            for item in data:
                # Transform each entry according to your defined transformation.
                transformed_item = {
                    "instruction": new_instruction,
                    "input": f"Claim: {item.get('claim', '')}\n\nOverall understanding:\n{item.get('understanding', '')}",
                    "output": item.get('label', '')
                }
                transformed_data.append(transformed_item)

            # Create new file name by adding "transform_" prefix.
            output_file_name = "transform_" + filename
            output_file_path = os.path.join(subdir, output_file_name)

            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    json.dump(transformed_data, f, indent=4, ensure_ascii=False)
                print(f" Transformed {input_file_path} and saved to {output_file_path} (Total points: {len(transformed_data)})")
            except Exception as e:
                print(f" Could not save {output_file_path}: {e}")
