# Multilingual — X-Fact, RU22Fact

TBE-3 applied across 25 languages (X-Fact, 7 classes) and 4 languages (RU22Fact, 3 classes).
Produces Table 9.

Neither dataset ships evidence text — only claim URLs — so this section has an extra step 0
that crawls, chunks and retrieves before the entailment pipeline starts.

```
0_preprocessing/           crawl → chunk → FAISS/e5 retrieval → jsonl
1_evidence_classification/ step 1 — SUPPORT / REFUTE per chunk
2_justification_generation/ step 2 — supporting + refuting justification
3_train_elm/               step 3a — XLM-R / mBERT + QLoRA
4_qlora_glm/               step 3b — 4-bit GLM + QLoRA via LLaMA-Factory
5_evaluation/              significance tests
data/justifications/       step-2 outputs, committed for both datasets × 4 GLMs
```

## Models

| Role | Model |
|---|---|
| ELM | `xlm-roberta-large`, `bert-base-multilingual-cased` |
| GLM (4-bit) | `unsloth/mistral-7b-instruct-v0.2-bnb-4bit`, `unsloth/llama-3-8b-Instruct-bnb-4bit`, `unsloth/gemma-7b-it-bnb-4bit`, `unsloth/Qwen2-7B-bnb-4bit` |
| Retrieval | `intfloat/multilingual-e5-large-instruct` + FAISS |
| Sentence splitting | spaCy `xx_sent_ud_sm`, falling back to blingfire, then NLTK |

Seeds: default, 26, 144 (the `*_lora_sft.yaml`, `*_lora_sft26.yaml`, `*_lora_sft144.yaml`
config triples).

## Running it

Environment: `entail-vllm` for steps 0–3 and inference, `entail-llamafactory` for step 4's
training.

### Step 0 — evidence enrichment

X-Fact and RU22Fact give claim URLs. We crawl each with Crawl4AI, split the page into
semantic chunks, and keep the single chunk that best matches the claim:

```bash
python 0_preprocessing/document_chunking.py    # semantic chunking of the crawled pages
python 0_preprocessing/retriever.py            # FAISS over multilingual-e5, top-1 chunk
```

Writes `data/processed/semantic_chunking/{xfact,ru22fact}/*.jsonl`.

`sentence_split.py` handles the 25 languages; `dataset_statistic.py` reports per-language
counts and evidence coverage.

A sentence-level chunking variant exists in the code paths. **The paper uses semantic
chunking** — that is the variant whose justifications are committed here.

### Steps 1–2 — entailment and consolidation

Both scripts carry their model list and dataset paths in a dict at the top of the file;
uncomment the model and split you want:

```bash
python 1_evidence_classification/evidence_classification.py
python 2_justification_generation/justification_generation.py
```

Output lands in `output_of_evidence_classification/` and
`output_of_justification_generation/<dataset>/<model>/`, mirroring the committed layout under
`data/justifications/`.

X-Fact splits are `train`, `Indomain`, `ood`, `zeroshot` (the paper's Table 1 "test" figure
is these three summed). RU22Fact has `train` and `test`.

### Step 3a — XLM-R / mBERT + QLoRA

```bash
bash 3_train_elm/slm_training_xfact_xlmr/train_xfact.sh
```

Edit the variables at the top of the shell script to switch GLM justifications or dataset:

```bash
DATA_DIR=../data/justifications/xfact/mistral
TRAIN_FILE=train_semantic_chunking_justifications.json
VAL_FILE=ood_semantic_chunking_justifications.json      # X-Fact validates on OOD
TEST_FILE=Indomain_semantic_chunking_justifications.json # and tests on in-domain
MODEL_NAME=xlm-roberta-large                             # or bert-base-multilingual-cased
LORA_R=16; LORA_ALPHA=32; LR=1e-4; EPOCHS=20; PATIENCE=4
MIN_EPOCHS=12; MIN_F1=0.22                               # 0.30 for RU22Fact
```

`MIN_EPOCHS` / `MIN_F1` stop the run early-stopping out at a degenerate macro-F1 — X-Fact's
7-way, 25-language setting needs the floor.

The three directories differ only in these defaults; the trainer
(`train_xfact_entialment.py`) is the same program.

### Step 3b — GLM + QLoRA

```bash
# convert justifications to LLaMA-Factory alpaca format
python 4_qlora_glm/prepare_data/to_llamafactory.py \
    --dataset_type xfact \
    --input_dir  data/justifications \
    --output_dir converted_datasets/xfact
# (--dataset_type ∈ xfact | ru22fact | lair_raw | rawfc;
#  --single_file <path> converts just one file)

# register + train
bash ../env/link_llamafactory.sh
cd $LLAMAFACTORY_DIR
llamafactory-cli train examples/entailment/multilingual/xfact/mistral_lora_sft.yaml

# inference (back in entail-vllm)
bash 4_qlora_glm/run_inference.sh
```

`run_inference.sh` sets `BASE_MODEL`, `ADAPTER`, `DATA` and `MAX_LENGTH`
(2500 for X-Fact, 1000 for RU22Fact) and calls `inference_test.py`, which loads the base
model in 4-bit and applies the adapter with PEFT.

Note the QLoRA rows in Table 9 are same-model: each GLM is fine-tuned on justifications it
generated itself (`Gemma_Gemma`, `Llama_Llama`, `Mistral_Mistral`, `Qwen_Qwen`).

### Step 4 — evaluation

```bash
python 5_evaluation/significance_xfact.py
```
