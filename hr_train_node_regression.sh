#!/bin/sh
#SBATCH --job-name=hr_train_node
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000

nvidia-smi

. venv/bin/activate

srun python train.py --model 'GCN' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_gcn.log' --model_dir 'saved_models/hr/gcn' --seed 42
srun python train.py --model 'GAT' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_gat.log' --model_dir 'saved_models/hr/gat' --seed 42
srun python train.py --model 'GIN' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_gin.log' --model_dir 'saved_models/hr/gin' --seed 42
srun python train.py --model 'GraphSAGE' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_sage.log' --model_dir 'saved_models/hr/sage' --seed 42
srun python train.py --model 'NodeEdgeGNN' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_nodeedge.log' --model_dir 'saved_models/hr/nodeedge' --seed 42
srun python train.py --model 'NodeEdgeGNN_Dual' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_nodeedge_dual.log' --model_dir 'saved_models/hr/nodeedge_dual' --seed 42
srun python train.py --model 'SWEGNN' --config_path='configs/hr_config.yaml' --log_path 'logs/hr/hr_swegnn.log' --model_dir 'saved_models/hr/swegnn' --seed 42
