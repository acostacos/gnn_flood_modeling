#!/bin/sh
#SBATCH --job-name=lr_validate
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000

nvidia-smi

. venv/bin/activate

srun python validate.py --model 'GCN' --config_path 'configs/lr_config.yaml'  --output_dir 'saved_metrics/lr/gcn' -seed 42 --model_path 'saved_models/lr/gcn/model_path.pt'
srun python validate.py --model 'GAT' --config_path 'configs/lr_config.yaml'  --output_dir 'saved_metrics/lr/gat' -seed 42 --model_path 'saved_models/lr/gat/model_path.pt'
srun python validate.py --model 'GIN' --config_path 'configs/lr_config.yaml'  --output_dir 'saved_metrics/lr/gin' -seed 42 --model_path 'saved_models/lr/gin/model_path.pt'
