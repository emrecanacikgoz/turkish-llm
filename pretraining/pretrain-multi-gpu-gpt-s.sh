#!/bin/bash
#SBATCH --job-name= <your-job-name> # Isim tanimlayin
#SBATCH -p <your-partition-name> # Kuyruk ismi
#SBATCH -A <your-project-name> # Proje ismi
#SBATCH -o %J-<your-job-name>.o%j # Cikti dosyasi
#SBATCH --gres=gpu:8        # Kac adet GPU isteyecegini belirtin
#SBATCH -N 1                # Kac adet node isteyecegini belirtin
#SBATCH -n 1                # Kac adet gorev isteyecegini belirtin
#SBATCH --cpus-per-task 128 # Kac adet CPU isteyecegini belirtin
#SBATCH --time=3-0:0:0      # Sure siniri koyun

source activate <your-env-name> # Conda ortamini aktif edin

torchrun --standalone --nproc_per_node=8 train.py \
    --out_dir 'gpt-small' \
    --eval_interval 2000 \
    --log_interval 1 \
    --eval_iters 200 \
    --eval_only False \
    --wandb_log True \
    --wandb_project 'llm-scratch' \
    --wandb_run_name 'gpt-small' \
    --dataset 'culturax' \
    --gradient_accumulation_steps 40 \
    --batch_size 12 \
    --block_size 1024 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --dropout 0.0 \
    --learning_rate 6e-4 \
    --max_iters 600_000 \
    --weight_decay 1e-1 \
    --grad_clip 1.0 \
    --warmup_iters 2000 \
    --lr_decay_iters 600_000 \
    --min_lr 6e-5 \
    --compile=False \  
