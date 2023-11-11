#!/bin/bash
#SBATCH --job-name=deneme
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_a6000
#SBATCH --mem=64G
#SBATCH --time=7-0:0:0
#SBATCH --output=%J.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=eacikgoz17@ku.edu.tr

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

source activate nanogpt
echo 'number of processors:'$(nproc)
nvidia-smi

python train.py \
    --out_dir 'out-deneme' \
    --eval_interval 2000 \
    --log_interval 1 \
    --eval_iters 200 \
    --eval_only False \
    --wandb_log True \
    --wandb_project 'llm-scratch' \
    --wandb_run_name 'gpt2-125m' \
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
    --lr_decay_iters 600000 \
    --min_lr 6e-5 \
    --compile=False \  
