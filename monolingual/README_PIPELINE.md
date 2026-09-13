# Monolingual — LIAR-RAW and RAW-FC

This directory is the original LIAR-RAW / RAW-FC codebase, kept structurally as-is. The old
`README.md` beside this file is preserved unchanged for reference, **but its commands name
files that do not exist** — use the ones below instead. See `../docs/CHANGES.md` for the
full list of edits.

Environment: `entail-vllm` for generation and ELM training, `entail-llamafactory` for the
LoRA / LoRA+ GLM training.

---

## TBE-3 — training on entailed justifications (the paper's method, Table 4)

### LIAR-RAW

```bash
cd TBE3/lair_raw

# 1. entailment — classify each evidence sentence SUPPORT / REFUTE
python lair_raw_evidences_classification.py --data_dir ./first_step_data_lair_raw
#    add --limit 5 for a smoke run

# 2. consolidation — one supporting + one refuting justification per claim
python justification_generation.py

# 3. normalise the generated text
python cleaning_data.py --input_dir ./outputs --output_dir ./cleaned_data

# 4-5. veracity prediction
python roberta_large_lair_raw_tranning.py
python XLNET_large_lair_raw_tranning.py
```

### RAW-FC

```bash
cd TBE3/rawfc

python Rawfc_evidence_classification.py --output ./results --data_dir .
python generated_justifications.py
python cleaning_data.py --input_dir ./generated_justification_rawfc --output_dir ./cleaned_data
python roberta_traning_early_full_all.py
python Xlnet_traning_early_full_all.py
```

Use the `*_seed.py` / `*_with_seed.py` variants to reproduce the three-seed averages.
`TBE3/rawfc/final_data_for_justification/` already holds the step-1 outputs per GLM, so you
can enter at step 2.

### LoRA / LoRA+ on the generated justifications

```bash
cd TBE3/LoRA_and_LoRA+_finetuning
bash LIAR_RAW_LoRA.sh      # LIAR_RAW_LoRA++.sh, RAWFC_LoRA.sh, RAWFC_LoRA++.sh
```

Each script runs `llamafactory-cli train` over the five per-GLM YAMLs in
`examples/train_lora/<dataset>/`. Point `model_name_or_path`, `dataset` and `output_dir` at
your own paths there. The three-seed configs are in `examples/train_lora/random_seeds/`.

Inference on the trained adapters:

```bash
cd TBE3/LoRA_and_LoRA+_finetuning/inferences_files/lair_raw/LoRA
python lair_raw_inference.py --config configs/llama.json
cd ../../rawfc/LoRA
python rawfc_inference_LoRA.py --config configs/llama.json
```

---

## TBE-2 — training on claim-evidence understanding (Table 4)

```bash
cd TBE2

# 1. generate the GLM's overall understanding (shared by both datasets)
python 1_generate_understanding.py \
    --data_dir ../TBE3/rawfc/data_rawfc_step1 \
    --train transformed_data_train_rawfc.json \
    --val   transformed_data_val_rawfc.json \
    --test  transformed_data_test_rawfc.json \
    --models llama qwen2.5 gemma mistral falcon \
    --out_dir outputs/rawfc

# 2. veracity prediction from the understanding
python rawfc/2_roberta_train.py
python rawfc/2_xlnet_train.py

# 3. (optional) LoRA / LoRA+ path — convert to LLaMA-Factory format first
python llamafactory_format/transform_rawfc.py
```

LIAR-RAW uses the same step 1 with `--data_dir ../TBE3/lair_raw/first_step_data_lair_raw`
and its own filenames, then `lair_raw/2_roberta_train.py` / `2_xlnet_train.py`.

The `understanding_test_*.json` files for all five GLMs are already committed under
`../IBE2/` and `../IBE3/`, so you can skip step 1 for the test split.

---

## TBE-1 — training on raw evidence sentences (Table 3)

```bash
cd TBE1/lair_raw
bash run.sh            # LoRA on Llama / Mistral / Falcon
cd ../loraplus
python lair_raw_lora_plusmistral_flash_attention_nan_loss.py
```

ELMs are deliberately absent here: many claim+evidence inputs exceed RoBERTa's and XLNet's
1,024-token limit, which is the reason the paper restricts TBE-1 to GLMs with adapters.

---

## IBE-1 … IBE-4 — prompting baselines (Table 5)

| | LIAR-RAW | RAW-FC |
|---|---|---|
| IBE-1 zero-shot | `IBE1/lair_raw/ibe1_lair_raw.py` | `IBE1/rawfc/ibe1_rawfcall.py` |
| IBE-2 two-step | `IBE2/lair_raw/IBE2_lair_raw_zeroshot.py` | `IBE2/rawfc/ibe2_rawfc_zero_short.py` |
| IBE-3 CoT | `IBE3/lair_raw/IBE3_lair_raw_cot.py` | `IBE3/rawfc/IBE3_rawfc_cot.py` |
| IBE-4 entailment | `IBE4/lair_raw/IBE4_lair_raw.py` | `IBE4/rawfc/ibe3_rawfcall.py` |

Per-model variants (`*_llama.py`, `*_qwen.py`, `*_gemma.py`, `*_mistral.py`, `*_falcon.py`)
sit beside each entry point; the un-suffixed script runs the default model.

Each writes `<model>_generated_results.json` plus a `classification_report.txt`, both of
which are committed for every model so the table can be re-derived without a GPU.

IBE-2/3 read the committed `understanding_test_*.json`; IBE-4 reads the committed
`generated_justifications_test_*.json`. Neither needs its first step re-run.

---

## Layout

```
TBE1/                  LoRA / LoRA+ on raw evidence               (Table 3)
TBE2/                  understanding generation + ELM/GLM training (Table 4)
TBE3/                  entailment → justification → veracity       (Table 4)
  lair_raw/            5-step LIAR-RAW pipeline
  rawfc/               5-step RAW-FC pipeline
  LoRA_and_LoRA+_finetuning/   vendored LLaMA-Factory + this project's configs
IBE1..IBE4/            prompting baselines                         (Table 5)
```
