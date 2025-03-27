#!/bin/sh
#SBATCH --job-name=hr_preprocess
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=long
#SBATCH --mem-per-cpu=48000

. venv/bin/activate

srun python3 preprocess.py  --config_path 'configs/hr_preprocess_config.yaml' --debug True --log_path 'logs/hr_preprocess.log'
