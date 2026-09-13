# Multimodal — Factify-2, MOCHEG, VERITE

TBE-3 applied to image+text claims. Produces Table 8.

A vision GLM (VLM) does the entailment and consolidation over `(claim, claim image,
evidence, evidence image)`; the veracity predictor is either an encoder (RoBERTa /
XLNet / DeBERTa, "FINE-TUNING" rows) or a text GLM with QLoRA (Llama-3.1-8B /
Mistral-7B, "QLoRA" rows).

```
0_data_preparation/       split, attach labels, resolve MOCHEG image paths
1_evidence_classification/ step 1 — SUPPORT / REFUTE per evidence item
2_justification_generation/ step 2 — supporting + refuting justification
3_train_elm/              step 3a — RoBERTa / XLNet / DeBERTa + QLoRA
4_qlora_glm/              step 3b — Llama / Mistral + QLoRA via LLaMA-Factory
5_evaluation/             classification reports + significance tests
data/justifications/      step-2 outputs, committed for all three datasets
```

## Models

| Key | Model | Notes |
|---|---|---|
| `qwen2vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | full precision |
| `idefics3` | `leon-se/Idefics3-8B-Llama3-bnb_nf4` | 4-bit |
| `paligemma` | `google/paligemma2-3b-mix-448` (4-bit build for VERITE) | needs `enforce_eager=True` under vLLM |

Seeds: 42, 57, 196.

## Running it

Environment: `entail-vllm` for steps 0–3, `entail-llamafactory` for step 4's training.

### Step 1 — evidence classification

`evidence_classification_unified.py` handles all datasets from one entry point:

```bash
python 1_evidence_classification/evidence_classification_unified.py \
    --data_path    ../data/raw/factify2/train.csv \
    --dataset_type factify \
    --model_key    qwen2vl \
    --output_json  outputs/train_evidence_qwen2vl.json \
    --cache_dir    image_cache \
    --checkpoint_interval 500
```

`--dataset_type` ∈ `factify | mocheg | verite | mr2`. Images are downloaded once and cached
under `--cache_dir`; the run is resumable via the checkpoint interval.

PaliGemma needs its own script (it cannot batch under vLLM):

```bash
python 1_evidence_classification/evidence_classification_paligemma.py \
    --data_path ../data/raw/verite/VERITE.csv --dataset_type verite \
    --output_json outputs/verite_evidence_paligemma.json --batch_size 4
```

The per-dataset scripts (`evidence_classification_factify.py`,
`evidence_classification_mocheg.py`) are the earlier, narrower versions, kept for reference.

### Step 2 — justification generation

```bash
python 2_justification_generation/justification_generation_qwenvl_idefics3.py \
    --input_json  outputs/train_evidence_qwen2vl.json \
    --output_json outputs/train_justifications_qwen2vl.json \
    --model_key   qwen2vl --checkpoint_interval 1000

# normalise + drop over-length records -> *_final_justification.json
python 2_justification_generation/prepare_final_justifications.py
```

`prepare_final_justifications.py` is what produces the `*_final_justification.json` names that
step 3 expects. `token_count.py` reports the length distribution against the ELM's 512/1024
token budget.

### Attaching labels (MOCHEG, VERITE)

Factify-2 carries its label inline. MOCHEG and VERITE do not:

```bash
python 0_data_preparation/attach_labels.py \
    --justifications data/justifications/mocheg/qwen2vl/test_justifications.json \
    --labels        data/raw/mocheg/test/Corpus2.csv \
    --just_id_field Id --label_id_field claim_id --label_field cleaned_truthfulness \
    --output        test_justifications_with_labels.json
```

### Step 3a — encoder + QLoRA

```bash
bash 3_train_elm/run_factify_roberta.sh     # or run_factify_xlnet.sh / run_mocheg_*.sh
```

which calls, e.g.:

```bash
python 3_train_elm/train_elm_factify.py \
    --model_name roberta-large \
    --data_dir   ../data/justifications/factify2/qwen \
    --seeds 42 57 196 --batch_size 8 --epochs 20 \
    --use_qlora --lora_r 16 --lora_alpha 32 \
    --output_dir my_experiment_results
```

`--model_name` ∈ `roberta-large | xlnet-large-cased | microsoft/deberta-v3-large`.
`--min_epochs` / `--min_f1` guard against early-stopping on a degenerate first epoch.

Ablations (Table 6's multimodal counterpart):
`train_elm_verite_support_only.py`, `train_elm_verite_refute_only.py`,
`train_elm_mocheg_ablation.py`.

### Step 3b — GLM + QLoRA

```bash
# convert (entail-vllm or any python)
python 4_qlora_glm/prepare_data/to_llamafactory.py --dataset factify2 \
    --input  data/justifications/factify2/qwen/train_evidence_qwen2vl_justification_final_justification.json \
    --output train_llamafactory_qwen_factify.json

# register + train (entail-llamafactory)
bash ../env/link_llamafactory.sh
cd $LLAMAFACTORY_DIR
llamafactory-cli train examples/entailment/multimodal/factify2/qwen_data/llama3_lora_sft_bnb_npu_qwen_57.yaml

# inference (entail-vllm)
python 4_qlora_glm/inference/llama_factify_inference.py \
    --adapter_path saves/llama3-8b/factify/qwen/lora/sft/seed_57 \
    --test_file    test_llamafactory_qwen_factify.json \
    --output_file  predictions/factify_llama_qwen_57.json
```

Config naming: `configs/<dataset>/<vlm>_data/<glm>_lora_sft_bnb_npu_..._<seed>.yaml`.
Files starting `llama3_` fine-tune Llama-3.1-8B; those with `mistral` in the name fine-tune
Mistral-7B — the `llama3_` prefix on some Mistral configs is a naming leftover, so read the
`model_name_or_path` field rather than the filename.

### Step 4 — evaluation

```bash
python 5_evaluation/classification_all_factify_slm.py    # macro-F1 per model × seed
python 5_evaluation/significance_factify2.py             # the * markers in Table 8
python 5_evaluation/significance_verite.py
```
