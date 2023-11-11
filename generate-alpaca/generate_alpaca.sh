#!/bin/bash
#SBATCH --job-name=alpaca
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=10G
#SBATCH --time=7-0:0:0
#SBATCH --output=%J.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=eacikgoz17@ku.edu.tr

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo
module load anaconda/3.6
source activate turco
echo 'number of processors:'$(nproc)
nvidia-smi
cat /proc/cpuinfo | grep 'processor' | wc -l

python -m generate_instruction generate_instruction_following_data \
    --output_dir ./ \
    --num_instructions_to_generate 4000 \
    --model_name "gpt-4"  \
    --seed_tasks_path "/kuacc/users/eacikgoz17/el-turco/turkish-llm/generate-alpaca/seed_tasks.jsonl"

source deactivate

