#!/bin/bash

# ================================
# User-configurable variables
# ================================

DATA_DIR=../Entailment_based_factchecking/output_of_justification_generation/xfact/qwen
TRAIN_FILE=train_semantic_chunking_justifications.json
VAL_FILE=ood_semantic_chunking_justifications.json
TEST_FILE=Indomain_semantic_chunking_justifications.json

MODEL_NAME=bert-base-multilingual-cased
DATASET_ID=xfact_qwen
OUTPUT_DIR=./checkpoints

BATCH_SIZE=8
EPOCHS=20
PATIENCE=4
LR=1e-4

MIN_EPOCHS=12
MIN_F1=0.22

SEEDS="42"

LORA_R=16
LORA_ALPHA=32

# ================================
# Run training
# ================================

python train_xfact_entialment.py \
  --data_dir ${DATA_DIR} \
  --train_file ${TRAIN_FILE} \
  --val_file ${VAL_FILE} \
  --test_file ${TEST_FILE} \
  --dataset_id ${DATASET_ID} \
  --model_name ${MODEL_NAME} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --lr ${LR} \
  --min_epochs ${MIN_EPOCHS} \
  --min_f1 ${MIN_F1} \
  --seeds ${SEEDS} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --output_dir ${OUTPUT_DIR} \
  --use_qlora
