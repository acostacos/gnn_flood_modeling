# GCN
python validate.py --model 'GCN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gcn_validate.log' --output_dir 'saved_metrics/init/gcn' --seed 42 --model_path 'saved_models/init/gcn/GCN_for_initp01_2025-05-13_13-03-32.pt'
python validate.py --model 'GCN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gcn_validate.log'  --output_dir 'saved_metrics/init/gcn' --seed 42 --model_path 'saved_models/init/gcn/GCN_for_initp02_2025-05-13_13-03-32.pt'
python validate.py --model 'GCN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gcn_validate.log'  --output_dir 'saved_metrics/init/gcn' --seed 42 --model_path 'saved_models/init/gcn/GCN_for_initp03_2025-05-13_13-03-32.pt'
python validate.py --model 'GCN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gcn_validate.log'  --output_dir 'saved_metrics/init/gcn' --seed 42 --model_path 'saved_models/init/gcn/GCN_for_initp04_2025-05-13_13-03-32.pt'

# GAT
python validate.py --model 'GAT' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gat_validate.log'  --output_dir 'saved_metrics/init/gat' --seed 42 --model_path 'saved_models/init/gat/GAT_for_initp01_2025-05-13_13-05-14.pt'
python validate.py --model 'GAT' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gat_validate.log'  --output_dir 'saved_metrics/init/gat' --seed 42 --model_path 'saved_models/init/gat/GAT_for_initp02_2025-05-13_13-05-14.pt'
python validate.py --model 'GAT' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gat_validate.log'  --output_dir 'saved_metrics/init/gat' --seed 42 --model_path 'saved_models/init/gat/GAT_for_initp03_2025-05-13_13-05-14.pt'
python validate.py --model 'GAT' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gat_validate.log'  --output_dir 'saved_metrics/init/gat' --seed 42 --model_path 'saved_models/init/gat/GAT_for_initp04_2025-05-13_13-05-14.pt'

# GIN
python validate.py --model 'GIN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gin_validate.log' --output_dir 'saved_metrics/init/gin' --seed 42 --model_path 'saved_models/init/gin/GIN_for_initp01_2025-05-13_13-07-32.pt'
python validate.py --model 'GIN' --config_path 'configs/config.yaml'  --log_path 'logs/init/init_gin_validate.log' --output_dir 'saved_metrics/init/gin' --seed 42 --model_path 'saved_models/init/gin/GIN_for_initp02_2025-05-13_13-07-32.pt'
python validate.py --model 'GIN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gin_validate.log' --output_dir 'saved_metrics/init/gin' --seed 42 --model_path 'saved_models/init/gin/GIN_for_initp03_2025-05-13_13-07-32.pt'
python validate.py --model 'GIN' --config_path 'configs/config.yaml' --log_path 'logs/init/init_gin_validate.log' --output_dir 'saved_metrics/init/gin' --seed 42 --model_path 'saved_models/init/gin/GIN_for_initp04_2025-05-13_13-07-32.pt'
