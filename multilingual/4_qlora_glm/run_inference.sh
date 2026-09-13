#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# -------- CONFIG --------
# BASE_MODEL="unsloth/llama-3-8b-Instruct-bnb-4bit"
# BASE_MODEL="unsloth/mistral-7b-instruct-v0.2-bnb-4bit"
BASE_MODEL="unsloth/gemma-7b-it-bnb-4bit"
# BASE_MODEL="unsloth/Qwen2-7B-bnb-4bit"
ADAPTER="../LLaMA-Factory/saves/lairraw_gemma_train_langchain_chunking_with_justifications/qlora/sft"
DATA="../output_of_justification_generation/lair_raw/langchain_chunking/gemma/test_justifications.json"
OUT_DIR="../output_of_entailment"

# Optional arguments
MAX_EXAMPLES=""      # e.g. "--max_examples 100"
# MAX_LENGTH="2500"   # sequence length for xfact
MAX_LENGTH="1000"   # sequence length for ru22fact
# ------------------------

echo "Running X-FACT  evaluation"
echo "Base model : $BASE_MODEL"
echo "Adapter    : $ADAPTER"
echo "Data       : $DATA"
echo "Output dir : $OUT_DIR"
echo "--------------------------"

python test.py \
  --base_model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --out_dir "$OUT_DIR" \
  --max_length "$MAX_LENGTH" \
  $MAX_EXAMPLES

echo "Evaluation finished successfully."
