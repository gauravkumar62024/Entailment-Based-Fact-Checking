import json
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pathlib import Path
import argparse
import torch
from pathlib import Path

# ======================================
# MODELS (same names as classification)
# ======================================

MODELS = {
    # "unsloth/llama-3-8b-Instruct-bnb-4bit": "llama",
    # "unsloth/Qwen2-7B-bnb-4bit": "qwen",
    # "unsloth/gemma-7b-it-bnb-4bit": "gemma",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": "mistral"
}

# DATASETS from your classification script
DATASET_LIST = [
    # "train_semantic_chunking", # processing train semantic chunking on node 1(anushandhan) for mistral
    # "Indomain_semantic_chunking", 
    # "ood_semantic_chunking",
    # "zeroshot_semantic_chunking",

    # "train_sentence_chunking",
    # "Indomain_sentence_chunking",
    # "ood_sentence_chunking",     # processing ood sentence chunking on node 0(anushandhan) for mistral
    # "zeroshot_sentence_chunking"  # processing zeroshot sentence chunking on node 0(anushandhan) for mistral
    # ru22fact datasets
    # "train",
    "test",
]


# ======================================
# ARGUMENTS
# ======================================
parser = argparse.ArgumentParser()
parser.add_argument("--evidence_results_dir", required=True,
                    help="Folder containing evidence classifier outputs.")
parser.add_argument("--output", required=True,
                    help="Folder to save justification outputs.")
args = parser.parse_args()


# ======================================
# LOAD EVIDENCE CLASSIFICATION OUTPUT
# ======================================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ======================================
# MODEL SETUP
# ======================================
def setup_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        max_model_len=7000,
        gpu_memory_utilization=0.30
    )

    sampling = SamplingParams(
        temperature=0.001,
        top_p=0.9,
        max_tokens=300
    )

    return model, sampling, tokenizer


# ======================================
# JUSTIFICATION PROMPT
# ======================================

def build_justification_prompt(claim, language, supporting, refuting):
    """
    Build justification prompts depending on available evidence.
    Ensures all prompts follow the same format as the original supporting-IF prompt.
    """

    # ---------------------------------------------------------
    # Supporting justification prompt
    # ---------------------------------------------------------
    if supporting:
        support_text = " ".join([s['sentence'] for s in supporting])
        support_prompt = (
            f"Given a {language} claim: {claim}\n a veracity label True,"
            f"analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) "
            f"based on the provided evidence. Do not mention the label in your response. "
            f"Focus only on key supporting factors and logical reasoning.\n"
            f"Evidence:\n{support_text}"
        )
    else:
        support_prompt = (
            f"Given a {language} claim: {claim}\n a veracity label True,"
            f"analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer). "
            f"Since no evidence is provided, rely on logical inference and generally known facts. "
            f"Do not mention any label in your response. Focus on key factors that could support the claim.\n"
            # f"Evidence: None"
        )

    # ---------------------------------------------------------
    # Refuting justification prompt
    # ---------------------------------------------------------
    if refuting:
        refute_text = " ".join([s['sentence'] for s in refuting])
        refute_prompt = (
            f"Given a {language} claim: {claim}\n a veracity label False,"
            f"analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer) "
            f"based on the provided evidence. Do not mention the label in your response. "
            f"Focus only on key factors and logical reasoning that challenge or contradict the claim.\n"
            f"Evidence:\n{refute_text}"
        )
    else:
        refute_prompt = (
            f"Given a {language} claim: {claim}\n a veracity label False,"
            f"analyze the statement and provide a concise, well-reasoned explanation (150 words or fewer). "
            f"Since no evidence is provided, rely on logical inference and generally known facts. "
            f"Do not mention any label in your response. Focus on key factors that could refute the claim.\n"
            # f"Evidence: None"
        )

    return support_prompt, refute_prompt


# ======================================
# MODEL GENERATION
# ======================================
def generate_answer(model, tokenizer, sampling, prompt, model_id):

    if "gemma" in model_id.lower():
        user_msg = {"role": "user", "content": prompt}
        messages = [user_msg]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        system_msg = {
            "role": "system",
            "content": "You have been specially designed to perform abductive reasoning for the fake news detection task and justification generation."
        }
        user_msg = {"role": "user", "content": prompt}
        messages = [system_msg, user_msg]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    output = model.generate([text], sampling)
    return output[0].outputs[0].text.strip()

# ======================================
# TRUNCATION FUNCTION
# ======================================

def truncate_prompt(tokenizer, text, limit=6900):
    tokens = tokenizer.encode(text)
    if len(tokens) > limit:
        tokens = tokens[:limit]
    return tokenizer.decode(tokens)

# ======================================
# PROCESS ONE DATA SPLIT
# ======================================
def process_split(data, model, tokenizer, sampling, model_id):
    out = []

    for entry in tqdm(data, desc="Generating Justifications"):
        claim = entry["claim"]
        label = entry["label"]
        language = entry["language"]

        supporting = entry["supporting_sentences"]
        refuting = entry["refuting_sentences"]

        # Build prompts
        support_prompt, refute_prompt = build_justification_prompt(
            claim, language, supporting, refuting
        )
        support_prompt = truncate_prompt(tokenizer, support_prompt)
        refute_prompt = truncate_prompt(tokenizer, refute_prompt)

        # Generate text
        support_out = generate_answer(model, tokenizer, sampling, support_prompt, model_id)
        refute_out = generate_answer(model, tokenizer, sampling, refute_prompt, model_id)

        out.append({
            "claim": claim,
            "label": label,
            "support_justification": support_out,
            "refute_justification": refute_out,
            "language": language
        })

    return out


# ======================================
# MAIN
# ======================================
def main():
    evidence_root = Path(args.evidence_results_dir)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_path, short_name in MODELS.items():
        print(f"\n=== Processing model: {short_name} ===")
        dataset_type = "/".join(evidence_root.parts[-2:])
        model_out_dir = output_root / dataset_type / short_name
        model_out_dir.mkdir(parents=True,exist_ok=True)

        # Load model
        model, sampling, tokenizer = setup_model(model_path)

        for dataset_name in DATASET_LIST:
            evidence_file = evidence_root / short_name / f"{dataset_name}.json"

            if not evidence_file.exists():
                print(f"Skipping missing file: {evidence_file}")
                continue

            print(f"→ Loading {evidence_file}")
            data = load_json(evidence_file)
            # data = data[:2]
            # Generate justification
            justifications = process_split(data, model, tokenizer, sampling, model_path)

            # Save
            out_file = model_out_dir / f"{dataset_name}_justifications.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(justifications, f, indent=2, ensure_ascii=False)

            print(f"✔ Saved to: {out_file}")

        # Cleanup
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
