from vllm import LLM, SamplingParams
import pandas as pd

# Load your claims
with open("claims_only.txt", "r") as f:
    claims = [line.strip() for line in f if line.strip()]

# Initialize model and sampling
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.1, max_tokens=256)

# Define function to create conversation format for vLLM chat
def build_conversation(claim):
    return [
        {
            "role": "system",
            "content": (
                "You are a classification assistant trained to assign **only one** appropriate taxonomy label to factual claims "
                "using a fixed, predefined list of broad, high-level categories. "
                "Do NOT generate multiple categories, speculative answers, or unrelated content. "
                "Do NOT invent new categories. Your response must strictly follow the format."
            )
        },
        {
            "role": "user",
            "content": f"""
Please analyze the following factual claim and assign it to **only one** of the categories from the taxonomy list provided below. 
Select the best matching category based on the overall **topic and context** of the claim, not specific keywords. 
Do not guess or return unrelated or undefined categories.

Claim:
"{claim}"

Choose ONLY ONE from the following taxonomy categories:
- Fact-Based Information Seeking
- Clarification and Concept Explanation
- Health and Wellness
- Economics and Finance
- Technical Assistance and Problem Solving
- Legal and Regulatory Information
- Politics and Government
- Social and Political Issues
- Personal Lifestyle and Hobbies
- History and Culture
- Technology and Digital Support
- Science and Nature
- Entertainment and Media
- Crime and Safety
- Product and Shopping Queries
- Travel and Geography
- Conspiracy and Misinformation

 STRICT FORMAT:
Your response MUST be limited to the following format with no additional text:
<category> [Category Name] </category>
<explanation> [Brief, 1-2 sentence justification of why this category was chosen] </explanation>

Do NOT generate anything outside this format. Do NOT include multiple categories or commentary.
"""
        }
    ]


import json

results = []
for i, claim in enumerate(claims):
    print(f"⏳ Processing claim {i+1}/{len(claims)}...")
    conversation = build_conversation(claim)
    output = llm.chat(conversation, sampling_params=sampling_params, use_tqdm=False)
    response = output[0].outputs[0].text.strip()

    # Try parsing structured output
    try:
        category = response.split("<category>")[1].split("</category>")[0].strip()
        explanation = response.split("<explanation>")[1].split("</explanation>")[0].strip()
    except Exception:
        category, explanation = "Unclassified", response  # fallback

    results.append({
        "claim": claim,
        "category": category,
        "explanation": explanation,
        "llm_response": response  # Save raw output too
    })


# Save to JSON
with open("classified_claims_with_taxonomy.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("✅ Classification saved to classified_claims_with_taxonomy.json")

