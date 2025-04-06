python train.py --model 'GCN' --config_path='configs/config.yaml' --log_path 'logs/init/init_gcn.log' --seed 42
python train.py --model 'GAT' --config_path='configs/config.yaml' --log_path 'logs/init/init_gat.log' --seed 42
python train.py --model 'GIN' --config_path='configs/config.yaml' --log_path 'logs/init/init_gin.log' --seed 42
python train.py --model 'GraphSAGE' --config_path='configs/config.yaml' --log_path 'logs/init/init_sage.log' --seed 42
python train.py --model 'NodeEdgeGNN' --config_path='configs/config.yaml' --log_path 'logs/init/init_nodeedge.log' --seed 42
python train.py --model 'NodeEdgeGNN_Dual' --config_path='configs/config.yaml' --log_path 'logs/init/init_nodeedge_dual.log' --seed 42
python train.py --model 'NodeEdgeGNN_NoPassing' --config_path='configs/config.yaml' --log_path 'logs/init/init_nodeedge_nopass.log' --seed 42
python train.py --model 'SWEGNN' --config_path='configs/config.yaml' --log_path 'logs/init/init_swegnn.log' --seed 42
