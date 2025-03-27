#!/bin/sh
#SBATCH --job-name=lr_train_node
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1

nvidia-smi

. venv/bin/activate

srun python train.py --model 'GCN' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_gcn.log' --seed 42
srun python train.py --model 'GAT' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_gat.log' --seed 42
srun python train.py --model 'GIN' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_gin.log' --seed 42
srun python train.py --model 'GraphSAGE' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_sage.log' --seed 42
srun python train.py --model 'NodeEdgeGNN' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_nodeedge.log' --seed 42
srun python train.py --model 'SWEGNN' --config_path='configs/lr_train_config.yaml' --log_path 'logs/lr/lr_swegnn.log' --seed 42
