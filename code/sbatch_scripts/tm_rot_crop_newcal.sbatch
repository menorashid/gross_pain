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
conda activate gross_pain
python train_encdec_pose.py --config_file ${CONFIG_FILE} --dataset_path /local_storage/users/sbroome/treadmill/ --train_subjects ${TRAIN_SUBJECTS} --test_subjects ${TEST_SUBJECTS} --job_identifier treadmillRotCropNewCal --config_file_model configs/config_train_rotation_crop_newCal.py --epoch_encdec 50 --train_subjects_model ${TRAIN_SUBJECTS_MODEL} --test_subjects_model ${TEST_SUBJECTS_MODEL} --job_identifier_encdec withRotCropNewCalMaheen
