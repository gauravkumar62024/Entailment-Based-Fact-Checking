export CUDA_VISIBLE_DEVICES=0
python tranning_w_o_resizing3_mocheg.py \
    --model_name xlnet-large-cased \
    --data_dir "mocheg_justification/qwen2vl" \
    --train_file train_justifications_final.json \
    --val_file val_justifications_final.json \
    --test_file test_justifications_final.json \
    --seeds 42 57 196 \
    --batch_size 8 \
    --use_qlora \
    --epochs 20 \
    --output_dir "my_experiment_results/mocheng/xlnet"  
    
#42 57 196 
