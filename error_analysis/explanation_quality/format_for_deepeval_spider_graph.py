import json

# Paths to your files
TEST_MERGED_PATH = 'test_merged.json'
GENERATED_PATH   = 'generated_justifications_test_meta_llama.json'
OUTPUT_PATH      = 'llama_rawfc_fordeepeval.json'

# Load both JSON files
with open(TEST_MERGED_PATH, 'r', encoding='utf-8') as f:
    test_entries = json.load(f)

with open(GENERATED_PATH, 'r', encoding='utf-8') as f:
    gen_entries = json.load(f)

# Build a lookup from claim to generated justifications
gen_lookup = {
    entry['claim']: entry
    for entry in gen_entries
}

# Merge
merged = []
for test in test_entries:
    claim = test['claim']
    original_expl = test.get('explain', '')

    gen = gen_lookup.get(claim)
    if not gen:
        # No matching generated justifications; skip or handle as you like
        continue

    true_j = gen.get('true_justification', '').strip()
    false_j = gen.get('false_justification', '').strip()

    # Concatenate them, you can adjust the separator as needed
    actual_expl = f"{true_j}\n\n{false_j}"

    merged.append({
        'claim': claim,
        'actual_explanation': actual_expl,
        'expected_explanation': original_expl
    })

# Write out the merged results
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Merged explanations written to {OUTPUT_PATH}")

