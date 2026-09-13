# Paper → code map

Every table and figure in the paper, and the script that produces it.

---

## Main results

### Table 2 — baselines (HiSS, FactLLaMA, RAFTS, L-Defense)

Reported from the original papers. The L-Defense row marked `†` is our reproduction, run
from the authors' released code (not vendored here).

### Table 3 — TBE-1, training on raw evidence sentences

| | |
|---|---|
| Code | `monolingual/TBE1/lair_raw/`, `monolingual/TBE1/loraplus/` |
| Runner | `monolingual/TBE1/lair_raw/run.sh` |
| Method | LoRA / LoRA+ on the five GLMs, claim + raw evidence in, veracity out |
| Note | ELMs are excluded here — many inputs exceed their 1,024-token limit |

### Table 4 — TBE-2 and TBE-3

**TBE-2** (claim-evidence *understanding*):

```
monolingual/TBE2/1_generate_understanding.py     # step 1, shared by both datasets
monolingual/TBE2/{lair_raw,rawfc}/2_roberta_train.py
monolingual/TBE2/{lair_raw,rawfc}/2_xlnet_train.py
monolingual/TBE2/llamafactory_format/transform_*.py   # → LoRA / LoRA+ path
```

**TBE-3** (entailed justifications — the paper's method):

```
monolingual/TBE3/lair_raw/lair_raw_evidences_classification.py   # step 1  entailment
monolingual/TBE3/lair_raw/justification_generation.py            # step 2  consolidation
monolingual/TBE3/lair_raw/cleaning_data.py                       # step 3  normalise
monolingual/TBE3/lair_raw/roberta_large_lair_raw_tranning.py     # steps 4-5  veracity
monolingual/TBE3/lair_raw/XLNET_large_lair_raw_tranning.py
```

RAW-FC mirrors this in `monolingual/TBE3/rawfc/` (`Rawfc_evidence_classification.py`,
`generated_justifications.py`, `cleaning_data.py`, `roberta_traning_early_full_all.py`,
`Xlnet_traning_early_full_all.py`).

The `*_seed.py` / `*_with_seed.py` variants are the three-seed runs behind the averages.
The GLM rows come from `monolingual/TBE3/LoRA_and_LoRA+_finetuning/*.sh`.

Significance markers (`*`, `**`, `***`) come from paired tests over the per-seed prediction
files; see `multimodal/5_evaluation/significance_factify2.py` for the same procedure applied
to the multimodal tables.

### Table 5 — IBE-1 … IBE-4 (prompting)

| Experiment | Directory | Entry point |
|---|---|---|
| IBE-1 zero-shot | `monolingual/IBE1/` | `ibe1_lair_raw.py`, `ibe1_rawfcall.py` |
| IBE-2 two-step | `monolingual/IBE2/` | `IBE2_lair_raw_zeroshot.py`, `ibe2_rawfc_zero_short.py` |
| IBE-3 CoT | `monolingual/IBE3/` | `IBE3_lair_raw_cot.py`, `IBE3_rawfc_cot.py` |
| IBE-4 entailment | `monolingual/IBE4/` | `IBE4_lair_raw.py`, `ibe3_rawfcall.py` |

Per-model variants (`*_llama.py`, `*_qwen.py`, …) sit beside each entry point. The
`understanding_test_*.json` and `generated_justifications_test_*.json` inputs are committed,
so IBE-2/3/4 run without re-doing their first step.

### Table 6 — ablation (w/o supporting / refuting justification)

Drop the corresponding field before training with the TBE-3 scripts above. The multimodal
equivalents are explicit:

```
multimodal/3_train_elm/train_elm_verite_support_only.py
multimodal/3_train_elm/train_elm_verite_refute_only.py
multimodal/3_train_elm/train_elm_mocheg_ablation.py
```

### Table 7 — performance vs. evidence count

Segment the test set by `len(tokenized_sentences)` and score each bucket. Buckets used:
`0`, `1`, `2–5`, `6–20`, `21–50`, `>50` (LIAR-RAW); `4–5`, `6–10`, `11–20`, `21–50`, `>50` (RAW-FC).
Cells 24–54 of `error_analysis/notebooks/rawfc_error_analysi.ipynb`.

---

## Domain generalisation

### Table 8 — multimodal (Factify-2, MOCHEG, VERITE)

```
multimodal/1_evidence_classification/evidence_classification_unified.py   # step 1
multimodal/2_justification_generation/justification_generation_*.py       # step 2
multimodal/2_justification_generation/prepare_final_justifications.py     # step 2b
multimodal/3_train_elm/train_elm_{factify,mocheg,verite}.py               # FINE-TUNING rows
multimodal/4_qlora_glm/configs/{factify2,mocheg,verite}/                  # QLoRA rows
multimodal/5_evaluation/significance_{factify2,verite}.py                 # the * markers
```

Row labels map as: `RoBERTa-L_QwenVL` = RoBERTa-large trained on Qwen2.5-VL justifications;
`Llama-8B_Idefics3` = Llama-3.1-8B (QLoRA) trained on Idefics3 justifications. Seeds 42 / 57 / 196.

Baselines: Du et al. (2023) for Factify-2 (reproduced at `max_seq_len=128` for GPU memory),
Cekinel et al. (2025) for MOCHEG (**not** reproduced — ~37 GPU-hours/epoch; the published
figure is shown in parentheses), Papadopoulos et al. (2025) for VERITE.

### Table 9 — multilingual (X-Fact, RU22Fact)

```
multilingual/0_preprocessing/{document_chunking,retriever}.py     # crawl → chunk → retrieve
multilingual/1_evidence_classification/evidence_classification.py # step 1
multilingual/2_justification_generation/justification_generation.py
multilingual/3_train_elm/slm_training_xfact_xlmr/train_xfact.sh   # XLM-R rows
multilingual/3_train_elm/slm_training_xfact_mbert/train_xfact.sh  # mBERT rows
multilingual/3_train_elm/slm_training_ru22fact_xlmr/train_xfact.sh
multilingual/4_qlora_glm/configs/{xfact,ru22fact}/                # QLoRA rows, seeds default/26/144
multilingual/5_evaluation/significance_xfact.py
```

`XLMR_Mistral` = XLM-R-large trained on Mistral-generated justifications. The QLoRA rows are
same-model (`Gemma_Gemma`, `Llama_Llama`, …): the GLM is fine-tuned on its own justifications.

X-Fact validation uses the out-of-domain split and testing the in-domain split (see
`train_xfact.sh`); RU22Fact uses its test split for both.

---

## Appendices

### Table 11 / Figure 3 / Table 18 — explanation quality (Appendix C)

```
error_analysis/explanation_quality/format_for_deepeval_spider_graph.py   # pair generated vs. gold
error_analysis/explanation_quality/vllms_prompting_for_explannation_evalution.py
error_analysis/explanation_quality/vllms_prompting_for_explannation_evalution_all_qwen_data.py
error_analysis/explanation_quality/{lair_raw,raw_fc}_spider_chart.py     # Figure 3 radar charts
error_analysis/notebooks/spider_graph.ipynb
```

ROUGE-1/2/L, BLEU and BERTScore (Table 11) plus GLM-as-judge scoring on five dimensions —
informativeness, accuracy, readability, objectivity, logicality.
Inputs are committed under `error_analysis/data/{deepeval_inputs,subjective_scores}/`.

### Table 12 — inter-annotator agreement (Appendix E)

Fleiss' κ and Krippendorff's α over 40 human-annotated samples. Cells 55–57 of
`error_analysis/notebooks/rawfc_error_analysi.ipynb`.

### Tables 16, 17 — attention visualisations (Appendix D)

`error_analysis/notebooks/attention.ipynb` — top-25% most-attended tokens in the best
models, whose training scripts are kept verbatim at `error_analysis/best_models/`.

### Confusion matrices (all 7 experiments)

Cells 75+ of `error_analysis/notebooks/rawfc_error_analysi.ipynb`, over the prediction files
in `error_analysis/data/predictions_for_confusion_matrix/` (one per experiment: `TBE1_lora_123_qwen`,
`TB2_llama_lora_plus_42`, `TBE3_rawfc_predictions_llama_123`, `IBE1qwen`, `IBE2meta_llama`,
`IBE3understanding_test_qwen`, `IBE4qwen`).

### Claim taxonomy / cluster error analysis

```
error_analysis/claim_taxonomy/taxonomy_builder_vllms.py   # 17-way topical taxonomy via Llama-3.1
error_analysis/claim_taxonomy/parser_for_the_vllms.py     # parse the labels back out
error_analysis/claim_taxonomy/anlaysis.py                 # accuracy per topic cluster
error_analysis/notebooks/bertopic.ipynb                   # BERTopic clustering
```

---

## Hyperparameters (Table 13)

| | ELMs | GLMs |
|---|---|---|
| Learning rate | 2e-6 / 2e-5 / 1e-5 | 1e-5 / 1e-4 |
| Optimizer | AdamW, Adam | — |
| Batch size | 8, 16 | 1 (× 8 grad-accum) |
| Early-stopping patience | 2, 3 | — |
| Epochs | up to 20 (early stop) | 3 |
| LoRA rank | 16 (QLoRA sections) | 8 |
| Scheduler | — | cosine, 10% warmup |
| Precision | — | bf16 |
| Decoding temperature | — | 0.001 |

Hardware: one NVIDIA A100 80GB.
