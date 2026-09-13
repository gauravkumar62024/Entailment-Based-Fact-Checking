---
library_name: peft
license: other
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: justification_rawfc_model_wise
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# justification_rawfc_model_wise

This model is a fine-tuned version of [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) on the rawfc_train_with_justi_for_lafct_formate_llama dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1127

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.0246        | 2.7503 | 500  | 0.1119          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.6.0+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0