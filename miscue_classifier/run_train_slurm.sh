#!/bin/bash
#SBATCH --job-name=deploytr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:59:59
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --partition=a40
#SBATCH --qos=a40
set -eo pipefail
source /home/daplab/30006664/spack/linux-nehalem/miniconda3-24.7.1-pfw3umki5a6bht3kflyum226yawke6x3/etc/profile.d/conda.sh
conda activate deb_wav2vec
set -u
cd /home/daplab/30006664/github_segment_features/miscue_classifier
python _train.py
