NPROC_PER_NODE=$gpu_count \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=$gpus \
swift sft \
    --model $model_path \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset $data_path \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit $freeze_vit \
    --freeze_llm $freeze_llm \
    --freeze_aligner $freeze_aligner \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 1 \
    --eval_steps -1 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir $output_path \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --report_to swanlab \
    --swanlab_token $swanlab_token \
    --swanlab_project $swanlab_project \
    --swanlab_exp_name $swanlab_exp_name \
    --swanlab_mode $swanlab_mode