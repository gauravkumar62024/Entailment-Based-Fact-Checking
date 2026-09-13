import json
import re
from pathlib import Path
import argparse

def clean_text(text):
    """
    Clean text by removing HTML tags, fixing punctuation, spacing, and capitalization.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    
    # Remove markdown formatting (bold)
    text = re.sub(r'\*\*', '', text)
    
    # Remove special tokens
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'\|\s*$', '', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r'\s+:', ':', text)
    
    # Fix capitalization after periods
    text = re.sub(r'\.\s*([a-z])', lambda m: f". {m.group(1).upper()}", text)
    
    # Fix first letter capitalization
    text = re.sub(r'^\s*([a-z])', lambda m: m.group(1).upper(), text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def clean_json_data(input_file, output_file):
    """
    Load JSON data, clean it, and save to a new file.
    Maintains the original array format of the JSON.
    Keeps 'support_justification' and 'refute_justification' keys.
    """
    try:
        # Load the JSON data
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} entries from {input_file}")
        
        # Clean each entry
        cleaned_data = []
        for entry in data:
            cleaned_entry = {
                "claim": entry["claim"],
                "label": entry["label"],
                "support_justification": clean_text(entry["support_justification"]),
                "refute_justification": clean_text(entry["refute_justification"])
            }
            cleaned_data.append(cleaned_entry)
        
        # Save the cleaned data as an array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully cleaned and saved {len(cleaned_data)} entries to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing justification files (e.g., ./results/generated_justification_rawfc_data)")
    parser.add_argument('--output_dir', type=str, required=True, help="Base output directory for cleaned files (e.g., ./cleaned_data)")
    args = parser.parse_args()

    # Define models and datasets
    MODELS = ["falcon", "mistral", "qwen", "llama", "gemma"]
    DATASETS = ["train", "test", "val"]

    input_base = Path(args.input_dir)
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        model_output_dir = output_base / model_name
        model_output_dir.mkdir(exist_ok=True)
        
        for split in DATASETS:
            input_file = input_base / model_name / f"generated_justification_{split}_{model_name}_rawfc.json"
            output_file = model_output_dir / f"{split}_{model_name}_cleaned.json"
            
            if not input_file.exists():
                print(f"Input file not found: {input_file}")
                continue
            
            print(f"Processing {input_file}")
            clean_json_data(input_file, output_file)

if __name__ == "__main__":
    main()
