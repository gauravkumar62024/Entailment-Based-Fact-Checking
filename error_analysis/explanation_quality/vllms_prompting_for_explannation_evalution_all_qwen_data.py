import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch, gc

# === Model List (short_key : huggingface_name) ===
models = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct-1M",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon": "tiiuae/Falcon3-7B-Instruct",
    "gemma": "google/gemma-7b-it"
}

# === Sampling Configuration ===
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=256,
)

# === Prompt Template ===
def make_prompt(actual, expected):
    return f"""You are an expert evaluator.

Your task is to evaluate a model-generated explanation ("generated") against a reference explanation ("gold") across five specific criteria.

For each criterion, assign an integer score from 1 to 5 (inclusive):
- 1 = very poor
- 2 = poor
- 3 = average
- 4 = good
- 5 = excellent

**Instructions:**
- Only return a **strict JSON object** in the exact format shown below.
- Do NOT include any explanation, comments, or additional text.
- Do NOT use floating point numbers or non-integer values.
- If you are unsure, always pick the conservative score.

**Criteria:**
1. Informativeness – Does it add meaningful context or background?
2. Accuracy – Does it reflect the correct meaning based on the reference?
3. Readability – Is it grammatically correct and easy to understand?
4. Objectivity – Is it neutral and free of emotional language?
5. Logicality – Does the reasoning follow a coherent and sound process?

**Return JSON (strict format):**
{{
  "Informativeness": int,
  "Accuracy": int,
  "Readability": int,
  "Objectivity": int,
  "Logicality": int
}}

Expected explanation:
\"\"\"{expected}\"\"\"

Actual explanation:
\"\"\"{actual}\"\"\"
"""

# === Load Dataset Once ===
with open("qwen_rawfc_fordeepeval.json") as f:
    data = json.load(f)

# === Loop over Models ===
for short_key, model_name in models.items():
    print(f"\n🚀 Running evaluation with model: {model_name} ({short_key})")

    # --- Load Model with Safe GPU Settings ---
    llm = LLM(
        model=model_name,
        max_model_len=4500,
        gpu_memory_utilization=0.9
    )

    results = []

    for idx, entry in enumerate(tqdm(data, desc=f"Evaluating with {short_key}")):
        prompt = make_prompt(entry["actual_explanation"], entry["expected_explanation"])

        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            json_start = generated_text.find("{")
            json_end = generated_text.find("}", json_start) + 1
            score_json = json.loads(generated_text[json_start:json_end])

            if not all(k in score_json for k in ["Informativeness", "Accuracy", "Readability", "Objectivity", "Logicality"]):
                raise ValueError("Missing keys in score JSON")

            result = {
                "index": idx,
                "claim": entry.get("claim", ""),
                "label": entry.get("label", ""),
                "actual": entry["actual_explanation"],
                "expected": entry["expected_explanation"],
                "scores": score_json,
                "model_response": generated_text,
                "error": None
            }

        except Exception as e:
            # Save the error entry with null score
            result = {
                "index": idx,
                "claim": entry.get("claim", ""),
                "label": entry.get("label", ""),
                "actual": entry["actual_explanation"],
                "expected": entry["expected_explanation"],
                "scores": None,
                "model_response": generated_text if 'generated_text' in locals() else "",
                "error": str(e)
            }

        results.append(result)

    # === Save Output ===
    output_file = f"qwen_{short_key}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Completed {short_key.upper()} → Saved {len(results)} entries → File: {output_file}")

    # === Cleanup GPU Memory ===
    del llm
    torch.cuda.empty_cache()
    gc.collect()

