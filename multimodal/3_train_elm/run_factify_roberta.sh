export CUDA_VISIBLE_DEVICES=0
python tranning_w_o_resizing3.py \
    --model_name roberta-large \
    --data_dir "outputs/justification/qwen" \
    --seeds 42 57 196 \
    --batch_size 8 \
    --use_qlora \
    --epochs 20 \
    --output_dir "my_experiment_results"  
    
#42 57 196 
