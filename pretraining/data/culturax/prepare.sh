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

source activate llm_tr_pretrain # Conda ortamini aktif edin

python prepare.py


