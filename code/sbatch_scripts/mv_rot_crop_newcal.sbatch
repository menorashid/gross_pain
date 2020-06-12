#!/usr/bin/env bash
#SBATCH --mem  6GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "rivendell|belegost|shire|khazadum|gondor"
#SBATCH --mail-type FAIL,END
#SBATCH --mail-user sbroome@kth.se
#SBATCH --output /Midgard/home/%u/projects/gross_pain/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/projects/gross_pain/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate tsnecuda
cd projects/gross_pain/code
python train_encode_decode.py --config_file ${CONFIG_FILE} --dataset_path ${DATASET_PATH} --train_subjects ${TRAIN_SUBJECTS} --test_subjects ${TEST_SUBJECTS} --job_identifier withRotCropNewCal
