# Datasets — where to get them, where to put them

Seven datasets are used. LIAR-RAW and RAW-FC are committed in full. The other five ship
images or crawled web pages that are too large for the repository, so only the **generated
justifications** are committed — enough to reproduce every number in Tables 8 and 9 without
re-running the GLM/VLM stages. Get the raw data only if you want to re-run steps 1–2.

---

## Committed in full

### LIAR-RAW (6 classes, English) · 10,065 / 1,274 / 1,251
### RAW-FC (3 classes, English) · 1,612 / 200 / 200

Both come from Yang et al. (2022), *A Coarse-to-fine Cascaded Evidence-Distillation Neural
Network for Explainable Fake News Detection* (COLING), and are Apache 2.0. We use their
original splits.

Already in the repo:

```
monolingual/TBE3/lair_raw/first_step_data_lair_raw/   # step-0 input, LIAR-RAW
monolingual/TBE3/rawfc/data_rawfc_step1/              # step-0 input, RAW-FC
monolingual/TBE3/rawfc/final_data_for_justification/  # step-1 output, per GLM
monolingual/IBE*/                                     # understandings + justifications
```

Nothing to download.

---

## Download required

Place each dataset at the path shown, or pass your own path on the command line — every
script takes the data location as an argument.

### Factify-2 (5 classes, English) · 27,500 / 7,500 / 7,500

- Source: <https://aiisc.ai/defactify2/> (De-Factify 2 @ AAAI-23, Suryavardan et al. 2023)
- Registration required.
- **Note on splits:** the official test set was never released. Following Cekinel et al.
  (2025) we sampled an equally sized validation set out of train and treat the official
  validation set as test. `multimodal/0_data_preparation/split_json.py` reproduces this.

```
multimodal/data/raw/factify2/
├── train.csv          # claim, claim_image, document, document_image, label
├── val.csv
└── test.csv           # = official validation split
```

### MOCHEG (3 classes, English) · 11,669 / 1,490 / 2,442

- Source: <https://github.com/VT-NLP/Mocheg> (Yao et al. 2023, SIGIR)
- Ships `Corpus2.csv` per split plus an image directory.

```
multimodal/data/raw/mocheg/
├── train/{Corpus2.csv, images/}
├── val/{Corpus2.csv, images/}
└── test/{Corpus2.csv, images/}
```

Gold labels live in `Corpus2.csv` (`cleaned_truthfulness`), not in the justification files.
Join them before training:

```bash
python multimodal/0_data_preparation/attach_labels.py \
    --justifications multimodal/data/justifications/mocheg/qwen2vl/test_justifications.json \
    --labels        multimodal/data/raw/mocheg/test/Corpus2.csv \
    --just_id_field Id --label_id_field claim_id --label_field cleaned_truthfulness \
    --output        test_justifications_with_labels.json
```

Image paths are attached separately with `mocheg_getting_image_path.py` →
`mocheg_add_image_path.py` (verify the join first with `mocheg_ids_match.py`).

### VERITE (3 classes, English) · 811 / 101 / 102

- Source: <https://github.com/stevejpapad/image-text-verification> (Papadopoulos et al. 2024)
- Run their `prepare_datasets.py` to materialise `VERITE.csv` and the image folder.

```
multimodal/data/raw/verite/
├── VERITE.csv         # columns: <index>, caption, image_path, label
└── images/
```

Labels join by the CSV's unnamed index column:

```bash
python multimodal/0_data_preparation/attach_labels.py \
    --justifications multimodal/data/justifications/verite/verite_full_justifications_qwen2vl.json \
    --labels        multimodal/data/raw/verite/VERITE.csv \
    --just_id_field claim_id --label_id_field "Unnamed: 0" --label_field label \
    --output        qwen2vl_verite_full_with_labels.json
```

### X-Fact (7 classes, 25 languages) · 19,079 / 2,535 / 9,575

- Source: <https://github.com/utahnlp/x-fact> (Gupta and Srikumar 2021)
- The test split aggregates the in-domain, out-of-domain and zero-shot evaluation sets; the
  pipeline keeps them separate (`Indomain` / `ood` / `zeroshot`).

```
multilingual/data/raw/xfact/
├── train.tsv
├── dev.tsv
├── in_domain.tsv
├── out_of_domain.tsv
└── zeroshot.tsv
```

**Evidence enrichment.** X-Fact ships claim URLs, not evidence text. We crawl each URL with
[Crawl4AI](https://github.com/unclecode/crawl4ai), segment the page into chunks, and keep the
single top claim-matching chunk retrieved by FAISS over `multilingual-e5-large-instruct`
embeddings. Run in order:

```bash
python multilingual/0_preprocessing/document_chunking.py   # semantic chunking
python multilingual/0_preprocessing/retriever.py           # FAISS + e5 top-chunk
```

This writes `multilingual/data/processed/semantic_chunking/xfact/{train,Indomain,ood,zeroshot}.jsonl`,
which is what steps 1–2 read. A sentence-level chunking variant also exists in the code; the
paper's numbers use **semantic** chunking, and only those justifications are committed.

### RU22Fact (3 classes, 4 languages) · 11,217 / 1,600 / 3,216

- Source: <https://github.com/ZhenyuMa/RU22Fact> (Zeng et al. 2024, LREC-COLING)
- Same crawl → chunk → retrieve preprocessing as X-Fact.

```
multilingual/data/raw/ru22fact/{train,val,test}.json
```

We could **not** reproduce the RU22Fact baseline (official code unavailable); Table 9 reports
the originally published figure for it.

---

## What ships instead of raw data

```
multimodal/data/justifications/
├── factify2/{qwen,idefics,paligemma}/{train,val,test}_*_final_justification.json
├── mocheg/{qwen2vl,idefices3,palegemma}/{train,val,test}_justifications.json
└── verite/…
multilingual/data/justifications/
├── xfact/{mistral,llama,gemma,qwen}/{train,Indomain,ood,zeroshot}_semantic_chunking_justifications.json
└── ru22fact/{mistral,llama,gemma,qwen}/{train,test}_semantic_chunking_justifications.json
```

These are the step-2 outputs. With them you can run step 3 (ELM training) and step 4 (GLM
QLoRA) directly and land on the paper's numbers, without any GPU time spent on generation.

`samples/` holds 20-record slices of each of these for smoke testing.

---

## Not committed

- Model checkpoints (`*.safetensors`, `*.pth`, `*.bin`) — regenerate by training, or ask the
  authors.
- Downloaded image caches (`cached_images/`, `image_cache/`) — rebuilt automatically on the
  first run of step 1.
- The `_with_tokens` and `_over512` intermediate files from the Factify-2 token-budget
  analysis — regenerate with `multimodal/2_justification_generation/token_count.py`.
- X-Fact **sentence**-level chunking justifications (the paper uses semantic chunking).
