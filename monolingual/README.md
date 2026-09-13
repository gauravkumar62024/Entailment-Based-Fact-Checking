# Entailment_base_fact_checking
## TBE3 
This experiment follows a few step to classify evidence, generate justifications, and fine-tune models using RoBERTa and XLNet. The overall idea is:
-   **Step 1:** Classify supporting and non-supporting evidence from the input.
    
-   **Step 2:** Use the classified evidence to generate textual justifications.
    
-   **Step 3:** Clean the generated justifications to standardize the input format.
    
-   **Step 4–5:** Train RoBERTa and XLNet models on the processed justifications.
### LAIR-RAW

```
# Step 1: Evidence Classification
cd TBE3/lair_raw
python 1_step_lair_raw.py --data_dir=./first_step_data_lair_raw
#Step 2: Generate Justifications
python justification_generation.py
#Step 3: Clean Justifications
python cleaning_data.py --input_dir=./outputs --output_dir=./cleaned_data
#Step 4: RoBERTa Training
python roberta_large_lair_raw_tranning_wo_seed.py
#Step 5: XLNet Training
python XLNET_large_lair_raw_tranning_wo.py
```
### RAWFC
```
# Step 1: Evidence Classification
cd TBE3/rawfc
python Step1_rawfc.py --output=./results
#Step 2: Generate Justifications
python generate_justifications.py --evidence_dir=./results/outputs --output=./generated_justification_rawfc
#Step 3: Clean Justifications
python cleaning_data.py --input_dir=./generated_justification_rawfc/generated_justification_rawfc_data --output_dir=./cleaned_data
#Step 4: RoBERTa Training
python roberta_traning_early_full_all_wo_seed.py
#Step 5: XLNet Training
python xlnet_large_traning_early_full_all_wo_seed.py

 ```
### LoRA and LoRA+ tranning on the generated justification.
This section describes how to fine-tune VLLMs using the generated justifications with LoRA and LoRA+ methods.
Each shell script (LIAR_RAW_LoRA.sh, RAWFC_LoRA.sh, etc.) automatically starts fine-tuning using configuration files located in folders such as example/train_lora/lair_raw. These configuration files allow you to customize:
 -   Model checkpoints path
    
-   Training data path
    
-   Base model name
    
-   Other training hyperparameters
You can modify the configurations to match your dataset or experimental settings.


## LAIR_RAW
```
cd TBE3/LoRA_and_LoRA+_finetuning
bash LIAR_RAW_LoRA.sh
bash LIAR_RAW_LoRA++.sh
```
## RAWFC

```
cd TBE3/LoRA_and_LoRA+_finetuning
RAWFC_LoRA.sh
RAWFC_LoRA++.sh
```

### Infernce on test data trained checkpoints
After training, the saved fine-tuned checkpoints are used for inference. Run the following command to evaluate the model on test data.
This script loads the trained LoRA model and performs inference on the test dataset, using settings specified in the configuration JSON file (e.g., model path, test data path, and decoding parameters).
```
cd LoRA_and_LoRA+_finetunning/inferences_files/rawfc/LoRA
python rawfc_inference_LoRA.py --config configs/llama.json
cd LoRA_and_LoRA+_finetunning/inferences_files/lair_raw/LoRA
python lair_raw_inference.py --config configs/llama.json
```
Note: The structure for training and inference remains the same for TBE2 and TBE3 (TB1). Only the configuration files and data paths change accordingly.

## TBE2 
### LAIR-RAW
```
cd TBE2/lair_raw
python run_LAIR_pipeline.py
```
### RAWFC
```
cd TBE2/rawfc
python run_rawfc_pipeline.py
```
### LoRA and LoRA+ tranning on the overall understanding.
## LAIR_RAW
```
cd TBE2/LoRA_LoRA+
bash LIAR_RAW_LoRA.sh
bash LIAR_RAW_LoRA++.sh
```
## RAWFC
```
bash RAWFC_LoRA.sh
bash RAWFC_LoRA++.sh
```
### Infernce on test data trained checkpoints
```
cd TBE2/LoRA_LoRA+/inferences_files/lair_raw/LoRA

python tbe2_lair.py --config configs/llama.json
cd TBE2/LoRA_LoRA+/inferences_files/rawfc/LoRA
python tbe2_rawfc.py --config configs/llama.json
```

## TBE1
### LAIR-RAW
```
cd TBE1/LoRA_LoRA_plus
bash LIAR_RAW_LoRA.sh
bash LIAR_RAW_LoRA++.sh
```
### RAWFC
```
cd TBE1/LoRA_LoRA_plus
bash RAWFC_LoRA.sh
bash RAWFC_LoRA++.sh
```
### Infernce on test data trained checkpoints
```
cd TBE1/LoRA_LoRA_plus/inferences_files/lair_raw/LoRA
python tbe1_lair.py --config configs/llama.json
cd TBE1/LoRA_LoRA_plus/inferences_files/rawfc/LoRA
python tbe1_rawfc.py --config configs/llama.json
```

## IBE1
### LAIR-RAW


```
python ibe1_lair_raw.py
```
### RAWFC
```
python ibe1_rawfcall.py
```
## IBE2
### LAIR-RAW
```
python IBE2_lair_raw_zeroshot.py
```
### RAWFC
```
python ibe2_rawfc_zero_short.py
```

## IBE3
### LAIR-RAW
```
python IBE3_lair_raw_cot.py
```
### RAWFC
```
python IBE3_rawfc_cot.py
```
## IBE4
### LAIR-RAW
```
python IBE4_lair_raw.py
```
### RAWFC
```
python ibe3_rawfcall.py
```

