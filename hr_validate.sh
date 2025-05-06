#!/bin/sh
#SBATCH --job-name=hr_validate
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000

nvidia-smi

. venv/bin/activate

srun python validate.py --model 'GCN' --config_path 'configs/hr_config.yaml'  --output_dir 'saved_metrics/hr/gcn' -seed 42 --model_path 'saved_models/hr/gcn/model_path.pt'
srun python validate.py --model 'GAT' --config_path 'configs/hr_config.yaml'  --output_dir 'saved_metrics/hr/gat' -seed 42 --model_path 'saved_models/hr/gat/model_path.pt'
srun python validate.py --model 'GIN' --config_path 'configs/hr_config.yaml'  --output_dir 'saved_metrics/hr/gin' -seed 42 --model_path 'saved_models/hr/gin/model_path.pt'
