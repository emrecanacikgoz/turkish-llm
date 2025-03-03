torchrun --nproc_per_node=8 finetune.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir path/to/output \
    --data-path alpaca-tr.json \
    --batch_size 16 \
    --micro_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1\
    --learning_rate 1e-4 \
    --cutoff_len 4096 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --group_by_length False \
    --prompt_template_name alpaca \
    --lr_scheduler 'cosine' \
    --warmup_steps 100