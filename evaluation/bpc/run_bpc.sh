#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --exclude=ai13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --constraint=rtx_a6000
#SBATCH --mem=40G
#SBATCH --time=7-0:0:0
#SBATCH --output=logs/%J-xglm-564M-2048.log
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

source /datasets/NLP/setenv.sh 
python bpc.py -m facebook/xglm-564M -d trnews-64 -w 2048

source deactivate


