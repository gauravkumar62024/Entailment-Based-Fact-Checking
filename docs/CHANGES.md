# Provenance and changes

This repository assembles code that lived in five working directories. This file records
where each part came from, what was changed, and what is knowingly missing — so the diff
against the original working copies is auditable.

---

## Where each section came from

| Section | Origin |
|---|---|
| `monolingual/` | `Entailment_base_fact_checking-main/` (the original public repo) |
| `monolingual/TBE1/` | `retrieve_entailemt/lair_raw/{lora_llms,loraplus}/` |
| `monolingual/TBE2/` | `retrieve_entailemt/rawfc/{exp3_rawfc,exp3_lair_raw,understanding}/` |
| `multimodal/` | `multimodalfctchecking/multimodal_fact_checking/` (+ the `/data2` twin for VERITE/MR2 inference) |
| `multilingual/` | `Babu/Entailment_based_factchecking/` and `Babu/slm_training_*` |
| `error_analysis/` | `retrieve_entailemt/rawfc/error_analysis_rawfc/` and `explanation_lair_raw_rawfc/` |

Where the same file existed in two working directories, the newer/more complete copy was
taken; for the multimodal inference scripts that is the `/data2` copy (it alone has the
VERITE and MR2 variants).

---

## Changes to `monolingual/` (kept deliberately minimal)

The original repo's structure is untouched. Six edits, all of which are bug fixes:

1. **`TBE3/lair_raw/lair_raw_evidences_classification.py`** — `data=data[:2]` (leftover debug
   truncation: the script only ever processed 2 claims) replaced with an opt-in
   `--limit N` flag, default 0 = full dataset.

2. **`TBE3/rawfc/Rawfc_evidence_classification.py`** — same fix for `data=data[:5]`. Also
   added `--data_dir` so the hard-coded relative `data_rawfc_step1/` path can be overridden,
   and `--models` / `--splits` so a single GLM or split can be run without loading all five
   models sequentially.

3. **`TBE3/lair_raw/lair_raw_evidences_classification.py`** (second issue) — four of the five GLMs were
   commented out in the `MODELS` list, so as shipped the script could only ever produce
   Qwen's evidence classification, not the five-model results the paper reports. All five are
   restored, keyed by short name, and selectable with the same `--models` / `--splits` flags.

4. **`IBE1/lair_raw/ibe1_lair_raw_gemma.py`** — `data=data[:8]` replaced with an optional
   `SMOKE_LIMIT` environment variable.

5. **`IBE3/lair_raw/IBE3_lair_raw_cot.py:232`** — syntax error
   (`target_names target_names` → `target_names=target_names`). The file could not be
   imported at all before this.

6. **`TBE3/LoRA_and_LoRA+_finetuning/data/dataset_info.json`** — registered two datasets that
   the Falcon YAMLs reference but that were never declared:
   `lair_raw_train_with_justi_for_lafct_formate_train_falcon3` and
   `rawfc_train_with_justi_for_lafct_formate_falcon3`. Both data files were already present;
   only the registration was missing, so the Falcon runs would have failed to start.

Removed (exact byte-for-byte duplicates, verified by md5):

- `TBE3/LoRA_and_LoRA+_finetuning/data/abc/` (120 MB — 4 files identical to their parents)
- `TBE3/LoRA_and_LoRA+_finetuning/data/rawfc_llamafact/` (24 MB — same)

Excluded from the copy: `*.safetensors`, `*.bin`, `*.pth`, `*.npy`, `checkpoint-*/`,
`__pycache__/`. These are trained artefacts, not source.

**The README's commands were wrong in the original repo** — every TBE-3 command named a file
that does not exist (`1_step_lair_raw.py`, `Step1_rawfc.py`, `generate_justifications.py`,
`*_wo_seed.py`, `*_wo.py`). The corrected commands are in the root `README.md`; the original
`monolingual/README.md` is left in place unchanged for reference.

---

## New code written for this repository

Four scripts were written because the originals were hard-coded to one dataset or one path,
which made the pipeline unrunnable as published:

| File | Replaces | Why |
|---|---|---|
| `monolingual/TBE2/1_generate_understanding.py` | `exp3_rawfc/rawfc_all_evidence_understand.py` | Same logic and prompt, but with `--data_dir/--train/--val/--test/--models/--limit` so it serves both LIAR-RAW and RAW-FC instead of only RAW-FC |
| `multimodal/4_qlora_glm/prepare_data/to_llamafactory.py` | `llama_factory_data_transformation_{factify,mocheng}.py` | The originals hard-coded input paths at module level and covered only two of the three datasets; this one adds VERITE and a CLI. The originals are kept beside it. |
| `multimodal/0_data_preparation/attach_labels.py` | ad-hoc joins | MOCHEG and VERITE keep gold labels in their own corpus files; this joins them onto the justifications by id |
| `smoke_test.sh` | — | Verifies the checkout without a GPU |

Everything else is copied verbatim.

---

## Verified

Stage 1 of `smoke_test.sh` (compile / parse / config-resolution / data transforms) passes on
both the development machine and a second machine with a clean checkout.

The GPU stage was verified end to end on an NVIDIA RTX A6000 with
`meta-llama/Llama-3.1-8B-Instruct`: RAW-FC step-1 entailment over 3 test claims classified
all 223 evidence sentences into coherent supporting / refuting groups in 45 s.

One environment prerequisite surfaced during that run and is now documented in the README,
checked by `env/setup.sh`, and auto-worked-around by `smoke_test.sh`: vLLM JIT-compiles
Triton kernels at startup and fails without `Python.h` (`python3-dev`). The fallback is
`export VLLM_USE_V1=0`.

---

## Known gaps

These are things the paper reports that this repository does not fully cover. Listed so a
reader is not misled.

### 1. Gold labels appear in the generation prompts

`monolingual/TBE3/lair_raw/lair_raw_evidences_classification.py:67` puts
`**Label for Claim:** {label}` into the evidence-classification prompt, and the script runs
over **train, val and test**. `justification_generation.py:133-143` interpolates the label
again (`Do not mention the label '{label}'`). So the LIAR-RAW features the ELM is evaluated
on were produced with access to the gold label.

RAW-FC's step 2 (`generated_justifications.py:89`) says "Do not mention the label" without
interpolating it — that stage is clean — but its step 1
(`Rawfc_evidence_classification.py:67`) does pass `{label}`.

This is left as-is because it is what produced the published numbers. Anyone extending the
work should remove it and re-run.

### 2. PaliGemma justification repair (MOCHEG)

`roberta_data/mocheg_justification_hari/modifier.py` in the original working directory
replaces a random 60% of PaliGemma's degenerate ("garbage") generations with evidence pulled
from Idefics3 and Qwen2-VL. It is **not** included in this pipeline and is not described in
the paper. MOCHEG PaliGemma rows in Table 8 should be read with this in mind.

### 3. Not reproduced baselines

- RU22Fact (Zeng et al. 2024) — official code unavailable.
- MOCHEG (Cekinel et al. 2025) — ~37 GPU-hours per epoch; the published figure is reported
  in parentheses in Table 9.

### 4. Confidence scores are computed and discarded

Step 1 prompts the GLM for `SUPPORT: 0.75` / `REFUTE: 0.60` and parses the float, but only
the binary label routes the sentence — the score is never used. This is true in all three
sections. It is a latent knob, not a bug.

### 5. Step 1 generates 1000 tokens for a 3-token answer

Both monolingual classifiers set `max_tokens=1000` in `SamplingParams`, but the prompt asks
for exactly `SUPPORT: 0.75` — about three tokens. The multilingual implementation uses
`max_tokens=5` for the same prompt. Lowering it is safe and makes step 1 dramatically faster
(it is the dominant cost of reproducing TBE-3); it is left at 1000 here because that is what
produced the published numbers, and generation is greedy (`temperature=0.01`) so the extra
budget does not change the parsed label.

### 6. Parsing fallbacks bias toward REFUTE

`extract_label_and_score()` returns `("REFUTE", 0.0)` when the model's reply does not match
the expected pattern. The RAW-FC variant instead drops the sentence entirely and logs it.
The two behave differently on malformed output.

### 7. Not carried over

- **MR2** — a fourth multimodal dataset with a complete pipeline in the working directories,
  but not part of the paper. Its code and configs are in the original directories if needed.
- **FEVER** — exploratory retrieval work (BM25 / Contriever / DPR / E5-Mistral) in
  `retrieve_entailemt/` and `multimodal_fact_checking/fever/`. This is the open-domain
  retrieval direction named as future work in the paper's Limitations, not part of the paper.
- **X-Fact sentence-level chunking** — an alternative to the semantic chunking the paper
  uses. Code supports it; only the semantic-chunking justifications are committed.
- **`SLMS_w_o_ocr.py`** — an ELM trainer with an auxiliary InfoNCE loss between the claim and
  support segments. It is not the trainer behind Table 8 (that is
  `3_train_elm/train_elm_factify.py`) and the loss is not described in the paper, so it is
  not included.
