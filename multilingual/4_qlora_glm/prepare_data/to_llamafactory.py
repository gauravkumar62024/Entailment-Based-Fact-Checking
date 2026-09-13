import json
import argparse
import os
from pathlib import Path

INSTRUCTIONS = {
    "ru22fact": (
        "Classify the given {language} claim into one of three categories "
        "(SUPPORTED, REFUTED, NEI) based on the provided true and false justifications:\n\n"
        "1. SUPPORTED: Fully supported by the context.\n"
        "2. REFUTED: Contradicted or unsupported by the context.\n"
        "3. NEI: Not enough information to verify.\n\n"
        "Provide exactly one label."
    ),

    "xfact": (
        "Classify the given {language} claim into one of the seven categories "
        "(TRUE, MOSTLY-TRUE, PARTLY-TRUE/MISLEADING, FALSE, MOSTLY-FALSE, "
        "COMPLICATED/HARD-TO-CATEGORIZE, OTHER) based on the provided true and false justifications.\n\n"
        "1. TRUE: The claim is fully supported by the support (true) justification, and the refute (false) justification does not meaningfully contradict it.\n"
        "2. MOSTLY-TRUE: The claim is largely supported by the support justification, but the refute justification highlights minor inaccuracies, limitations, or missing details.\n"
        "3. PARTLY-TRUE/MISLEADING: The support justification confirms some aspects of the claim, but the refute justification reveals significant omissions, distortions, or potentially misleading elements.\n"
        "4. FALSE: The refute (false) justification clearly contradicts or disproves the claim, and the support justification does not provide credible backing.\n"
        "5. MOSTLY-FALSE: The refute justification shows the claim is largely incorrect, though the support justification may contain a small element of truth. \n"
        "6. COMPLICATED/HARD-TO-CATEGORIZE: Too complex or nuanced to assign a straightforward label.\n"
        "7. OTHER: Does not fit into any of the above categories.\n\n"
        "Provide exactly one label."
    ),
    "lair_raw": (
    "Classify the given {language} claim into one of the six categories "
    "(TRUE, MOSTLY-TRUE, HALF-TRUE, BARELY-TRUE, FALSE, PANTS-FIRE) based on the provided true, false, and mixed justifications.\n\n"
    "1. TRUE: The claim is fully supported by the evidence with no significant inaccuracies.\n"
    "2. MOSTLY-TRUE: The claim is largely correct but contains minor inaccuracies or missing details.\n"
    "3. HALF-TRUE: The claim is partially correct but leaves out important information or presents facts in a misleading way.\n"
    "4. BARELY-TRUE: The claim contains a small element of truth but is mostly inaccurate or misleading.\n"
    "5. FALSE: The claim is not supported by the evidence or is directly contradicted.\n"
    "6. PANTS-FIRE: The claim is completely false and outrageously misleading.\n\n"
    "Provide exactly one label."
),
"raw_fc": (
    "Classify the given {language} claim into one of the three categories "
    "(TRUE, HALF-TRUE, FALSE) based on the provided true and false justifications.\n\n"
    "1. TRUE: The claim is fully supported by the evidence.\n"
    "2. HALF-TRUE: The claim is partially supported but includes significant omissions, distortions, or misleading elements.\n"
    "3. FALSE: The claim is not supported or is contradicted by the evidence.\n\n"
    "Provide exactly one label."
)

}

# Define file structure based on your image
FILE_STRUCTURE = {
    "ru22fact": {
        "gemma": [
            "train_semantic_chunking_justifications.json"
        ],
        "qwen": ["train_semantic_chunking_justifications.json"],
        "llama": ["train_semantic_chunking_justifications.json"],
        "mistral": ["train_semantic_chunking_justifications.json"],
    },
    "xfact": {
        "gemma": [
            "train_semantic_chunking_justifications.json",
            "train_sentence_chunking_justifications.json",
        ],
        "llama": [
            "train_semantic_chunking_justifications.json",
            "train_sentence_chunking_justifications.json",
        ],
        "mistral": [
            "train_semantic_chunking_justifications.json",
            "train_sentence_chunking_justifications.json",
        ],
        "qwen": [
            "train_semantic_chunking_justifications.json",
            "train_sentence_chunking_justifications.json",
        ]
    },
    "lair_raw": {
        "custom_chunking":{
            "gemma": [
            "train.json"
        ],
        "llama": [
             "train.json"
        ],
        "mistral": [
             "train.json"
        ],
        "qwen": [
             "train.json"
        ]
        },
        "langchain_chunking":{
            "gemma": [
            "train.json"
        ],
        "llama": [
             "train.json"
        ],
        "mistral": [
             "train.json"
        ],
        "qwen": [
             "train.json"
        ]
        }
    },
    "rawfc": {
        "custom_chunking":{
            "gemma": [
            "train.json"
        ],
        "llama": [
             "train.json"
        ],
        "mistral": [
             "train.json"
        ],
        "qwen": [
             "train.json"
        ]},
        "langchain_chunking":{
            "gemma": [
            "train.json"
        ],
        "llama": [
             "train.json"
        ],
        "mistral": [
             "train.json"
        ],
        "qwen": [
             "train.json"
        ]}
    }
}

def build_input(example):
    claim = example.get("claim", "").strip()
    support_justification = example.get("support_justification", "").strip()
    refute_justification = example.get("refute_justification", "").strip()

    return (
        f"Claim: {claim}\n\n"
        f"True Justification:\n{support_justification}\n\n"
        f"False Justification:\n{refute_justification}"
    )


def convert_dataset(input_path, output_path, dataset_type):
    if dataset_type not in INSTRUCTIONS:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    instruction_template = INSTRUCTIONS[dataset_type]
    alpaca_data = []

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for example in data:
        language = example.get("language", "English")
        label = example.get("label")

        alpaca_example = {
            "instruction": instruction_template.format(language=language),
            "input": build_input(example),
            "output": label.lower()   # ✅ KEEP LABEL EXACTLY AS DATASET
        }

        alpaca_data.append(alpaca_example)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(alpaca_data)} examples → {output_path}")


def process_all_files(dataset_type, base_input_dir="output_of_justification_generation", output_dir="converted_datasets"):
    """
    Process all files for a given dataset type from the specified folder structure
    """
    if dataset_type not in FILE_STRUCTURE:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(FILE_STRUCTURE.keys())}")
    
    dataset_path = Path(base_input_dir) / dataset_type
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory not found: {dataset_path}")
    
    processed_files = []
    
    # Process all models in the dataset
    for model in FILE_STRUCTURE[dataset_type]:
        model_path = dataset_path / model
        
        # Check if model directory exists
        if not model_path.exists():
            print(f"Warning: Model directory not found: {model_path}, skipping...")
            continue
        
        # Get list of files to process
        if FILE_STRUCTURE[dataset_type][model]:
            # Use predefined file list
            files_to_process = FILE_STRUCTURE[dataset_type][model]
        else:
            # If list is empty, process all JSON files in the directory
            files_to_process = [f.name for f in model_path.glob("*.json")]
        
        for filename in files_to_process:
            input_file = model_path / filename
            
            if not input_file.exists():
                print(f"Warning: File not found: {input_file}, skipping...")
                continue
            
            # Create output filename with model and dataset prefix
            output_filename = f"{dataset_type}_{model}_{filename}"
            output_file_path = output_path / output_filename
            
            # Convert the dataset
            convert_dataset(input_file, output_file_path, dataset_type)
            processed_files.append((input_file, output_file_path))
    
    return processed_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert dataset files to Alpaca format based on folder structure"
    )
    parser.add_argument(
        "--dataset_type",
        required=True,
        choices=["ru22fact", "xfact","lair_raw","rawfc"],
        help="Dataset type (ru22fact or xfact)"
    )
    parser.add_argument(
        "--input_dir",
        default="../output_of_justification_generation",
        help="Base input directory (default: output_of_justification_generation)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for converted files (default: converted_datasets)"
    )
    parser.add_argument(
        "--single_file",
        help="Process a single file instead of all files for the dataset type"
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        # Process a single file with manual input
        input_path = Path(args.single_file)
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model = input_path.parent.name
        chunking_type = input_path.parent.parent.name

        # base filename without extension
        base = input_path.stem.split("_justifications")[0]  # e.g., "train" OR "train_justifications"

        # remove existing "_justifications" if it already exists
        if base.endswith("_justifications"):
            base = base.replace("_justifications", "")

        output_filename = f"{args.dataset_type}_{model}_{base}_{chunking_type}_justifications.json"
        output_file_path = output_path / output_filename
        
        convert_dataset(input_path, output_file_path, args.dataset_type)
        print(f"Processed single file: {input_path} → {output_file_path}")
    else:
        # Process all files for the dataset type
        print(f"Processing all files for dataset: {args.dataset_type}")
        print(f"Input directory: {args.input_dir}/{args.dataset_type}")
        print(f"Output directory: {args.output_dir}")
        print("-" * 50)
        
        processed_files = process_all_files(args.dataset_type, args.input_dir, args.output_dir)
        
        if processed_files:
            print(f"\nSuccessfully processed {len(processed_files)} files:")
            for input_file, output_file in processed_files:
                print(f"  {input_file}")
                print(f"  → {output_file}")
                print()
        else:
            print(f"No files were processed for dataset type: {args.dataset_type}")


if __name__ == "__main__":
    main()