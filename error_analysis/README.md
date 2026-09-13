# Error analysis

Everything behind the paper's appendices and Figure 3. Notebook outputs are stripped (the
main notebook is 44 MB with outputs, 180 KB without) — re-run to regenerate the figures.

```
notebooks/            the four analysis notebooks
explanation_quality/  Appendix C — lexical, semantic and GLM-as-judge scoring
claim_taxonomy/       topical clustering of claims + per-cluster accuracy
best_models/          the exact training scripts behind the best LIAR-RAW / RAW-FC models
data/                 prediction files and scored explanations, committed
figures/              the PDFs that go into the paper
```

## Notebooks

| Notebook | Produces |
|---|---|
| `rawfc_error_analysi.ipynb` | macro-F1 / MP / MR bar charts per experiment, radar charts, confusion matrices for all 7 experiments, IAA (Table 12) |
| `attention.ipynb` | Appendix D — top-25% most-attended tokens (Tables 16, 17) |
| `spider_graph.ipynb` | Figure 3 — subjective evaluation radar charts |
| `bertopic.ipynb` | BERTopic clustering used by the claim taxonomy |

## Appendix C — explanation quality (Table 11, Figure 3, Table 18)

Two evaluations of the TBE-3 justifications treated as model explanations.

**Lexical + semantic.** ROUGE-1/2/L, BLEU, BERTScore against the gold explanations that ship
with LIAR-RAW and RAW-FC.

**GLM-as-judge.** Each GLM scores every GLM's explanations 1–5 on informativeness, accuracy,
readability, objectivity and logicality.

```bash
# 1. pair each generated justification with its gold explanation
python explanation_quality/format_for_deepeval_spider_graph.py

# 2. score with a judge model
python explanation_quality/vllms_prompting_for_explannation_evalution.py
python explanation_quality/vllms_prompting_for_explannation_evalution_all_qwen_data.py

# 3. radar charts (Figure 3)
python explanation_quality/lair_raw_spider_chart.py
python explanation_quality/raw_fc_spider_chart.py
```

Inputs are committed:
`data/deepeval_inputs/{llama,qwen,gemma,mistral,falcon}_{lair_raw,rawfc}_fordeepeval.json`
and the resulting per-judge scores in `data/subjective_scores/qwen_{model}.json`
plus `detailed_scores_*.csv` / `macro_scores_*.txt`.

## Claim taxonomy

Assigns each claim one of 17 topical categories with Llama-3.1-8B, then reports veracity
accuracy per category — this is where the "which claims does the model get wrong" analysis
comes from.

```bash
python claim_taxonomy/taxonomy_builder_vllms.py   # claims_only.txt -> classified_claims_with_taxonomy.json
python claim_taxonomy/parser_for_the_vllms.py     # regex the labels back out
python claim_taxonomy/anlaysis.py                 # -> cluster_veracity_error_analysis.csv
```

`anlaysis.py` reads `data/fixed_classified_claims1.json`, `data/predictions_llama_123.json`
and `data/misclassified_llama_123.json`, all committed.

## Confusion matrices

`data/predictions_for_confusion_matrix/` holds one prediction file per experiment:

| File | Experiment |
|---|---|
| `TBE1_lora_123_qwen_predictions_with_responses.json` | TBE-1 |
| `TB2_llama_lora_plus_42predictions_with_responses.json` | TBE-2 |
| `TBE3_rawfc_predictions_llama_123.json` | TBE-3 |
| `IBE1qwen_generated_results.json` | IBE-1 |
| `IBE2meta_llama_generated_results.json` | IBE-2 |
| `IBE3understanding_test_qwen.json` | IBE-3 |
| `IBE4qwen_generated_results.json` | IBE-4 |

Cells 75+ of `rawfc_error_analysi.ipynb` build `all_7_confusion_matrices.pdf` and the
combined per-dataset matrices from these. The notebook's paths point at the original working
directory — repoint them at `data/predictions_for_confusion_matrix/` before running.

## Note on the notebooks' paths

These notebooks were written against absolute paths under
`/data2/Gaurav/retrieve/rawfc/error_analysis_rawfc/`. They are committed as they were run;
adjust the paths in the first cell of each to point at `data/` here.
