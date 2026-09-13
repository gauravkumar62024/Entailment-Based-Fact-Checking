import json
import pandas as pd
import re

# -------------------------------
# Step 1: Load JSON
# -------------------------------
with open("classified_claims_with_taxonomy.json", "r", encoding="utf-8") as f:
    raw_results = json.load(f)

# -------------------------------
# Step 2: Regex Parsers
# -------------------------------


# Define your complete list of allowed taxonomy labels
TAXONOMY_LABELS = [
    "Fact-Based Information Seeking",
    "Clarification and Concept Explanation",
    "Health and Wellness",
    "Economics and Finance",
    "Technical Assistance and Problem Solving",
    "Legal and Regulatory Information",
    "Politics and Government",
    "Social and Political Issues",
    "Personal Lifestyle and Hobbies",
    "History and Culture",
    "Technology and Digital Support",
    "Science and Nature",
    "Entertainment and Media",
    "Crime and Safety",
    "Product and Shopping Queries",
    "Travel and Geography",
    "Conspiracy and Misinformation"
]

def extract_category(response: str) -> str:
    response_cleaned = re.sub(r"[\n\r]+", " ", response).strip().lower()

    # First, look for properly tagged format
    match = re.search(r"<category>(.*?)</category>", response, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for label in TAXONOMY_LABELS:
            if candidate.lower() in label.lower() or label.lower() in candidate.lower():
                return label

    # Second, look for tags like <Politics and Government> or <politics_and_government>
    tag_match = re.findall(r"<([a-zA-Z0-9_\- ]+)>", response_cleaned)
    for tag in tag_match:
        for label in TAXONOMY_LABELS:
            if tag.replace("_", " ").strip() in label.lower():
                return label

    # Third, match plain-text presence of label name inside the response
    for label in TAXONOMY_LABELS:
        if label.lower() in response_cleaned:
            return label

    return "Unclassified"

def extract_explanation(response: str) -> str:
    match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", response, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else response.strip()

# -------------------------------
# Step 3: Reprocess Entries
# -------------------------------
cleaned_results = []
for item in raw_results:
    claim = item.get("claim", "")
    response = item.get("llm_response", "")
    category = extract_category(response)
    explanation = extract_explanation(response)
    
    cleaned_results.append({
        "claim": claim,
        "category": category,
        "explanation": explanation,
        "llm_response": response
    })


with open("fixed_classified_claims1.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
print(" Saved: fixed_classified_claims.json")

