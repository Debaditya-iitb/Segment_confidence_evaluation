#!/bin/bash
#SBATCH --job-name=en_v2_train
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/daplab/30006664/github_en_miscue_v2/train/job.%j.train.out
#SBATCH --error=/home/daplab/30006664/github_en_miscue_v2/train/job.%j.train.err
set +u
source /home/daplab/30006664/spack/linux-nehalem/miniconda3-24.7.1-pfw3umki5a6bht3kflyum226yawke6x3/etc/profile.d/conda.sh
conda activate deb_wav2vec
set -u
python -u /home/daplab/30006664/github_en_miscue_v2/train/train_classifier.py
