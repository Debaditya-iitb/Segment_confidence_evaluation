#!/bin/bash
#SBATCH --job-name=en_v2_demo
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/daplab/30006664/github_en_miscue_v2/train/job.%j.demo.out
#SBATCH --error=/home/daplab/30006664/github_en_miscue_v2/train/job.%j.demo.err
set +u
source /home/daplab/30006664/spack/linux-nehalem/miniconda3-24.7.1-pfw3umki5a6bht3kflyum226yawke6x3/etc/profile.d/conda.sh
conda activate deb_wav2vec
set -u
R=/home/daplab/30006664/github_en_miscue_v2
cd $R

echo "=================== STEP 1 - features ==================="
python -u segment_features.py --lang en \
    --csv     demo/demo_input_english.csv \
    --out     demo/demo_english_features.csv \
    --model   /home/daplab/30006664/Wav2vec_models_Raj/xlsr_IITM_FT_WPP_5_FT_NSO_MID_END_CTC_LOWEST_TOP_40_hr \
    --matrix  matrices/confusion_english_KV_G345_new_foldavg.npz \
    --wav-scp demo/demo_english_wav.scp \
    --utt-col WavFileName

echo
echo "=================== STEP 2 - classify ==================="
python -u predict.py \
    --input  demo/demo_english_features.csv \
    --output demo/demo_english_scored.csv \
    --asr-label-col label \
    --slim
