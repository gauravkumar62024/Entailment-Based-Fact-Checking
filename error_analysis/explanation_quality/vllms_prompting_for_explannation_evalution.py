import json
from tqdm import tqdm
from vllm import LLM, SamplingParams

# === Step 1: Load vLLM Model ===
model_name = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(model=model_name)
sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.9,
    max_tokens=256,
)

# === Step 2: Prompt Template ===
def make_prompt(actual, expected):
    return f"""Evaluate the following model-generated explanation ("actual") by comparing it with the reference ("expected"). Rate each of the five criteria from 1 (poor) to 5 (excellent) and return a JSON object.

Criteria:
1. Informativeness: Does it add meaningful context or background?
2. Accuracy: Does it align with the expected explanation and true label?
3. Readability: Is it grammatically correct, coherent, and easy to understand?
4. Objectivity: Is it free from unnecessary emotional or subjective language?
5. Logicality: Does it follow a sound reasoning process that supports the conclusion?

Return JSON only in this format:
{{
  "Informativeness": x,
  "Accuracy": x,
  "Readability": x,
  "Objectivity": x,
  "Logicality": x
}}

Expected explanation:
\"\"\"{expected}\"\"\"

Actual explanation:
\"\"\"{actual}\"\"\"
"""

# === Step 3: Load Dataset ===
with open("llama_rawfc_fordeepeval.json") as f:
    data = json.load(f)

results = []

# === Step 4: Evaluate Each Entry ===
for idx, entry in enumerate(tqdm(data)):
    prompt = make_prompt(entry["actual_explanation"], entry["expected_explanation"])

    try:
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        json_start = generated_text.find("{")
        json_end = generated_text.find("}", json_start) + 1
        score_json = json.loads(generated_text[json_start:json_end])

        results.append({
            "index": idx,
            "claim": entry.get("claim", ""),
            "label": entry.get("label", ""),
            "actual": entry["actual"],
            "expected": entry["expected"],
            "scores": score_json,
            "model_response": generated_text
        })

    except Exception as e:
        print(f"Error at index {idx}: {e}\nGenerated:\n{generated_text}\n")
        continue

# === Step 5: Save Output ===
with open("llama_rawfc_eval_scores.json", "w") as f:
    json.dump(results, f, indent=2)

print("Justification evaluation complete. Saved to 'llama_rawfc_eval_scores.json'")

