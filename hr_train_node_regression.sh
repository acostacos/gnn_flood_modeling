python main.py --model 'GCN' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_gcn.log' --seed 42
python main.py --model 'GAT' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_gat.log' --seed 42
python main.py --model 'GIN' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_gin.log' --seed 42
python main.py --model 'GraphSAGE' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_sage.log' --seed 42
python main.py --model 'NodeEdgeGNN' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_nodeedge.log' --seed 42
python main.py --model 'SWEGNN' --config_path='configs/hr_train_config.yaml' --log_path 'logs/hr/hr_swegnn.log' --seed 42
