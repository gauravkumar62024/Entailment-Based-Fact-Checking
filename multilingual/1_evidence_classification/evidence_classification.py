import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams


# ============================================================
# CONFIGURATION
# ============================================================

MODELS = {
    # "unsloth/llama-3-8b-Instruct-bnb-4bit": "llama",
    # "unsloth/Qwen2-7B-bnb-4bit": "qwen",
    # "unsloth/gemma-7b-it-bnb-4bit": "gemma",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit": "mistral"
}

DATASETS = {
    # "train_semantic_chunking": "../Multilingual_factchecking/data/processed/semantic_chunking/xfact/train.jsonl",
    # "Indomain_semantic_chunking": "../Multilingual_factchecking/data/processed/semantic_chunking/xfact/Indomain.jsonl",
    # "ood_semantic_chunking": "../Multilingual_factchecking/data/processed/semantic_chunking/xfact/ood.jsonl",
    # "zeroshot_semantic_chunking": "../Multilingual_factchecking/data/processed/semantic_chunking/xfact/zeroshot.jsonl",

    # "train_sentence_chunking": "../Multilingual_factchecking/data/processed/sentence_chunking/xfact/train.jsonl",
    # "Indomain_sentence_chunking": "../Multilingual_factchecking/data/processed/sentence_chunking/xfact/Indomain.jsonl",
    # "ood_sentence_chunking": "../Multilingual_factchecking/data/processed/sentence_chunking/xfact/ood.jsonl",
    # "zeroshot_sentence_chunking": "../Multilingual_factchecking/data/processed/sentence_chunking/xfact/zeroshot.jsonl",
    # "liar_raw": "../data/processed/chunked_langchain/LIAR-RAW/train.json",
    "liar_raw": "../data/processed/chunked_langchain/LIAR-RAW/test.json",
    # "liar_raw": "../data/processed/chunked_custom/LIAR-RAW/train.json",
    # "liar_raw": "../data/processed/chunked_custom/LIAR-RAW/test.json",
    # "rawfc": "../data/processed/chunked_langchain/rawfc/train.json",
    # "rawfc": "../data/processed/chunked_langchain/rawfc/test.json"
    # "rawfc": "../data/processed/chunked_custom/rawfc/train.json"
}

LANGUAGE_MAP = {
    "tr": "Turkish", "ka": "Georgian", "pt": "Portuguese", "id": "Indonesian",
    "sr": "Serbian", "it": "Italian", "de": "German", "ro": "Romanian",
    "ta": "Tamil", "pl": "Polish", "hi": "Hindi", "ar": "Arabic", "es": "Spanish",
    "bn": "Bengali", "fa": "Persian", "gu": "Gujarati", "mr": "Marathi",
    "pa": "Punjabi", "no": "Norwegian", "si": "Sinhala", "sq": "Albanian",
    "ru": "Russian", "az": "Azerbaijani", "nl": "Dutch", "fr": "French"
}


# ============================================================
# MULTILINGUAL SENTENCE SPLITTING
# ============================================================

_nlp_multi = None

def load_sentence_splitter():
    global _nlp_multi
    if _nlp_multi is not None:
        return _nlp_multi

    import spacy
    import nltk
    from blingfire import text_to_sentences as bf_sent
    nltk.download("punkt", quiet=True)

    try:
        _nlp_multi = spacy.load("xx_sent_ud_sm")
    except:
        print("⚠ spaCy model missing; using fallback sentence splitters")
        _nlp_multi = False

    return _nlp_multi


from blingfire import text_to_sentences as bf_sent


def split_into_sentences(text: str) -> List[str]:
    nlp = load_sentence_splitter()

    if nlp:
        try:
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if sents:
                return sents
        except:
            pass

    try:
        sents = bf_sent(text).split("\n")
        return [s.strip() for s in sents if s.strip()]
    except:
        pass

    try:
        import nltk
        return nltk.sent_tokenize(text)
    except:
        return [text]


# ============================================================
# MODEL SETUP + TRUNCATION HELPERS
# ============================================================

FULL_PROMPT_MAX = 7500
EVIDENCE_MAX = 6000


def setup_model(model_path):
    model = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_model_len=8000,
        dtype="auto",
        gpu_memory_utilization=0.50
    )

    sampling = SamplingParams(
        temperature=0.0,
        top_p=0.95,
        max_tokens=5
    )

    return model, sampling


def truncate_tokens(tok, txt, max_toks):
    ids = tok.encode(txt)
    if len(ids) > max_toks:
        return tok.decode(ids[:max_toks])
    return txt


def enforce_prompt_limit(tok, prompt):
    ids = tok.encode(prompt)
    if len(ids) > FULL_PROMPT_MAX:
        return tok.decode(ids[-FULL_PROMPT_MAX:])
    return prompt


# ============================================================
# PROMPT GENERATION (MODEL-AGNOSTIC)
# ============================================================

SYSTEM_PROMPT = (
     "You are a {type} classifier. Given a claim and one evidence sentence, "
    "output exactly one word: SUPPORT or REFUTE. Do not include punctuation, scores, or any extra text."
)

USER_TEMPLATE = (
    "Classify the evidence sentence as SUPPORT or REFUTE relative to the {lang} claim.\n\n"
    "Claim: {claim}\n"
    "Evidence Sentence:\n{sentence}\n\n"
    "Output exactly one word: SUPPORT or REFUTE"
)
USER_TEMPLATE_GEMMA = (
    "You are a {type} classifier. Given a claim and one evidence sentence, "
    "output exactly one word: SUPPORT or REFUTE. Do not include punctuation, scores, or any extra text.\n\n"
    "Classify the evidence sentence as SUPPORT or REFUTE relative to the {lang} claim.\n\n"
    "Claim: {claim}\n"
    "Evidence Sentence:\n{sentence}\n\n"
    "Output exactly one word: SUPPORT or REFUTE"
)


def safe_apply_chat_template(tokenizer, messages):
    """
    Ensures that apply_chat_template() ALWAYS returns a string.
    """

    try:
        tmpl = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # -------------------------------
        # FIX for MISTRAL / MISTRAL_COMMON
        # -------------------------------
        # print("✔ Prompt generated via apply_chat_template()")
        if not isinstance(tmpl, str):
            tmpl = tokenizer.decode(tmpl)

        return tmpl
        
    except Exception:
        # print("⚠ apply_chat_template() failed; using manual prompt fallback.")
        # fallback manual prompt
        sys_msg = messages[0]["content"]
        user_msg = messages[1]["content"]
        return f"{sys_msg}\n\n{user_msg}\n\nAnswer:"


def build_prompts(claim,dataset_name, label, sentences, lang, tokenizer, model_id):
    prompts = []

    for sent in sentences:
        user_msg = USER_TEMPLATE.format(claim=claim, sentence=sent, lang=lang)
        if dataset_name.lower() in ["liar_raw", "rawfc"]:
            user_msg_for_gemma = USER_TEMPLATE_GEMMA.format(type="English evidence", claim=claim, sentence=sent, lang=lang)
        else:
            user_msg_for_gemma = USER_TEMPLATE_GEMMA.format(type="multilingual evidence", claim=claim, sentence=sent, lang=lang)
        if dataset_name.lower() in ["liar_raw", "rawfc"]:
            system_msg = SYSTEM_PROMPT.format(type="English evidence")
        else:
            system_msg = SYSTEM_PROMPT.format(type="multilingual evidence")
        if model_id.lower() == "gemma":
            messages = [{"role": "user", "content": user_msg_for_gemma}]
        else:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
        prompt = safe_apply_chat_template(tokenizer, messages)
        prompts.append(prompt)

    return prompts

def extract_retrieved_sentences(item):
    sentences = []

    for report in item.get("reports", []):
        for rc in report.get("retrieved_chunks", []):
            for sent in rc["chunk"].keys():
                sentences.append(sent)

    return sentences
# ============================================================
# CLASSIFICATION
# ============================================================

def classify_sentences(model, dataset_name, sampling, tokenizer, claim, label, evidences, lang, model_id, batch_size=128):

    supporting, refuting, all_judgments = [], [], []

    for ev in evidences:
        # Split evidence into sentences
        if dataset_name.lower() in ["liar_raw", "rawfc"]:
            clean_sents = [truncate_tokens(tokenizer, ev, 256)]
        else:
            sentences = split_into_sentences(ev)
            clean_sents = [truncate_tokens(tokenizer, s, 256) for s in sentences]

        # Build prompts
        prompts = [
            enforce_prompt_limit(tokenizer, p)
            for p in build_prompts(claim,dataset_name, label, clean_sents, lang, tokenizer,model_id)
        ]

        # Batch inference
        for i in range(0, len(prompts), batch_size):
            chunk_prompts = prompts[i:i+batch_size]
            chunk_sents   = clean_sents[i:i+batch_size]

            outputs = model.generate(chunk_prompts, sampling)

            for sent, out in zip(chunk_sents, outputs):

                # The raw model output
                raw = out.outputs[0].text.strip()
                clean = raw.upper()

                # -------------------------------------------------------
                # NEW: Robust regex-based extraction (no first-label restriction)
                # -------------------------------------------------------
                matches = re.findall(r"\b(SUPPORT|SUPPORTS|SUPPORTED|SUPPORTING|REFUTE|REFUTES|REFUTED|REFUTING)\b", clean)

                if len(matches) == 0:
                    pred = "UNKNOWN"
                else:
                    # If model produces multiple labels, choose the FINAL one
                    pred = matches[-1]
                # -------------------------------------------------------

                # Record judgment
                rec = {
                    "sentence": sent,
                    "raw_response": raw,
                    "prediction": pred
                }
                all_judgments.append(rec)

                if pred == "SUPPORT" or pred == "SUPPORTS" or pred == "SUPPORTED" or pred == "SUPPORTING":
                    supporting.append(rec)
                elif pred == "REFUTE" or pred == "REFUTES" or pred == "REFUTED" or pred == "REFUTING":
                    refuting.append(rec)

    return supporting, refuting, all_judgments




# ============================================================
# IO HELPERS
# ============================================================

def load_jsonl(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                out.append(json.loads(line))
        else:
            out = json.load(f)
    return out


# ============================================================
# DATASET PROCESSOR
# ============================================================

def process_dataset(dataset,split, model, tokenizer, sampling, dataset_name, model_short, out_dir):
    results = []

    for item in tqdm(dataset, desc=f"{model_short} → {dataset_name}"):
        if dataset_name.lower() == "liar_raw" or dataset_name.lower() == "rawfc":
            claim = item["claim"]
            label = item.get("label")
            lang  = "English"   # LIAR / RAWFC are English
            evidences = extract_retrieved_sentences(item)
        else:
            claim = item["claim"]
            label = item.get("label")
            lang = LANGUAGE_MAP.get(item.get("language", "").lower(), item.get("language", "").upper())
            evidences = [ev["evidence"] for ev in item.get("evidences", [])]

        supporting, refuting, judgments = classify_sentences(
            model,dataset_name, sampling, tokenizer,
            claim, label, evidences,
            lang, model_short 
        )

        results.append({
            "claim": claim,
            "label": label,
            "language": lang,
            "supporting_sentences": supporting,
            "refuting_sentences": refuting,
            "original_evidences": evidences,
            "sentence_judgments": judgments
        })

    out_path = out_dir / f"{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✔ Saved → {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    for model_path, short in MODELS.items():
        print(f"\n=== Loading model: {model_path} ===")
        model, sampling = setup_model(model_path)
        tokenizer = model.get_tokenizer()

        model_out = output_root / short
        model_out.mkdir(exist_ok=True)

        for dataset_name, dataset_path in DATASETS.items():
            print(f"\n--- Dataset: {dataset_name} ({dataset_path})")
            data = load_jsonl(dataset_path)
            split = Path(dataset_path).stem
            process_dataset(
                dataset=data,
                split=split,
                model=model,
                tokenizer=tokenizer,
                sampling=sampling,
                dataset_name=dataset_name,
                model_short=short,
                out_dir=model_out
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
