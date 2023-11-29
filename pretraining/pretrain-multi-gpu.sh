#!/bin/bash
#SBATCH --job-name=llm
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A proj12           # Kullanici adi
#SBATCH -o %J.out           # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:8        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 128  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=3-0:0:0      # Sure siniri koyun.

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

source activate nanogpt
echo 'number of processors:'$(nproc)
nvidia-smi

torchrun --standalone --nproc_per_node=8 train.py \
    --out_dir 'all-data-gpt-124m' \
    --eval_interval 2000 \
    --log_interval 1 \
    --eval_iters 200 \
    --eval_only False \
    --wandb_log True \
    --wandb_project 'llm-scratch' \
    --wandb_run_name 'all-data-gpt-124m' \
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
    --min_lr 1e-6 \
    --compile=False \  
