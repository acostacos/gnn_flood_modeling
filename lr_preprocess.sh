#!/bin/sh
#SBATCH --job-name=lr_preprocess
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=long

. venv/bin/activate

srun python3 preprocess.py  --config_path='configs/lr_preprocess_config.yaml' --log_path='logs/lr_preprocess.log'
