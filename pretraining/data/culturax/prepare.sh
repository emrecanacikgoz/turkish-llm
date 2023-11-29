#!/bin/bash
#SBATCH --job-name=deneme-llm
#SBATCH -p palamut-cuda     # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A proj12           # Kullanici adi
##SBATCH -J print_gpu       # Gonderilen isin ismi
#SBATCH -o %J.out           # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:4        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 64  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=3-0:0:0      # Sure siniri koyun.


echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
source activate nanogpt
echo 'number of processors:'$(nproc)
nvidia-smi

python prepare.py

#python check.py


