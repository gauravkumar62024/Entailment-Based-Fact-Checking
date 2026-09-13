# pip install pandas rouge-score nltk bert-score

import os
import json
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# 0) configure
TEST_FILE = 'test.json'
# put here all the generated files in /mnt/data or your cwd:
GENERATED_FILES = [
    'generated_justifications_test_falcon3.json',
    'generated_justifications_test_gemma.json',
    'generated_justifications_test_meta_llama.json',
    'generated_justifications_test_mistral.json',
    'generated_justifications_test_qwen.json',
]

# ensure NLTK tokenizer data
nltk.download('punkt')

# 1) load the reference data once
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
df_ref = pd.DataFrame(test_data)[['claim','explain']].rename(columns={'explain':'reference'})

# 2) prepare scorers
rouge = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1

for gen_file in GENERATED_FILES:
    # derive a short model name
    model_name = os.path.basename(gen_file).replace('generated_justifications_test_','')\
                                           .replace('.json','')
    print(f"\n⏳ Processing {model_name}…")
    
    # 3) load generated justifications
    with open(gen_file, 'r', encoding='utf-8') as f:
        gen_data = json.load(f)
    df_gen = pd.DataFrame(gen_data)[['claim','true_justification','false_justification']]
    
    # 4) merge on claim
    df = df_ref.merge(df_gen, on='claim', how='inner')
    
    # 5) build the candidate text
    df['candidate'] = df['true_justification'].str.strip() + ' ' + df['false_justification'].str.strip()
    
    refs  = df['reference'].tolist()
    cands = df['candidate'].tolist()
    
    # 6) compute BERTScore once for the batch
    P, R, F1 = bert_score(cands, refs, lang='en', rescale_with_baseline=True)
    
    # 7) compute per‑example metrics
    records = []
    for i, (ref, cand) in enumerate(zip(refs, cands)):
        r_scores = rouge.score(ref, cand)
        bleu = sentence_bleu([word_tokenize(ref)],
                             word_tokenize(cand),
                             smoothing_function=smooth)
        records.append({
            'rouge1_f': r_scores['rouge1'].fmeasure,
            'rouge2_f': r_scores['rouge2'].fmeasure,
            'rougeL_f': r_scores['rougeL'].fmeasure,
            'bleu':     bleu,
            'bert_p':   P[i].item(),
            'bert_r':   R[i].item(),
            'bert_f1':  F1[i].item()
        })
    scores_df = pd.DataFrame(records)
    
    # 8) macro‑average
    macro = scores_df.mean()
    print(f"→ {model_name} macro ROUGE‑1 F1: {macro['rouge1_f']:.4f}, "
          f"ROUGE‑2 F1: {macro['rouge2_f']:.4f}, ROUGE‑L F1: {macro['rougeL_f']:.4f}, "
          f"BLEU: {macro['bleu']:.4f}, BERT P: {macro['bert_p']:.4f}, "
          f"R: {macro['bert_r']:.4f}, F1: {macro['bert_f1']:.4f}")
    
    # 9) save outputs
    out_csv = f'detailed_scores_{model_name}.csv'
    out_txt = f'macro_scores_{model_name}.txt'
    scores_df.to_csv(out_csv, index=False)
    with open(out_txt, 'w') as out:
        out.write("metric\tvalue\n")
        for k,v in macro.items():
            out.write(f"{k}\t{v:.4f}\n")
    print(f"  • detailed → {out_csv}")
    print(f"  • macro    → {out_txt}")

