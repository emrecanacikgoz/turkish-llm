#!/bin/bash
#SBATCH --job-name=deneme
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
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


torchrun --nproc_per_node=2 --master_port=12345 finetune.py \
    --base_model mistralai/Mistral-7B-v0.1 \
    --data-path /kuacc/users/eacikgoz17/el-turco/data-tools/culturax-jsons/tr_part_00000-0.1GB.json \
    --output_dir ./mistral-7b-0.1GB-r8-a16-lr6e-4 \
    --mode pretrain \
    --batch_size 8 \
    --micro_batch_size 2 \
    --num_epochs 1 \
    --learning_rate 0.0006 \
    --cutoff_len 1024 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[gate_proj, down_proj, up_proj]' \
    --prompt_template_name culturax \
    --lr_scheduler 'cosine' \
    --warmup_steps 100 \
    --wand_run_name mistral-7b-0.1GB-r8-a16-lr6e-4 \

