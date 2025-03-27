import numpy as np
import traceback
import torch
import yaml

from argparse import ArgumentParser, Namespace
from typing import Tuple
from data import PreprocessFloodEventDataset
from models import GAT, GCN, GraphSAGE, GIN, MLP, NodeEdgeGNN, SWEGNN
from training import NodeRegressionTrainer, DualRegressionTrainer
from utils import Logger, file_utils
from utils.loss_func_utils import get_loss_func

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='configs/train_config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='NodeEdgeGNN', help='Model to use for training')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    return parser.parse_args()

def model_factory(model_name: str, **kwargs) -> Tuple[torch.nn.Module, str]:
    if model_name == 'NodeEdgeGNN':
        return NodeEdgeGNN(**kwargs), 'l1'
    if model_name == 'SWEGNN':
        return SWEGNN(**kwargs), 'l1'
    if model_name == 'GCN':
        return GCN(**kwargs), 'l1'
    if model_name == 'GAT':
        return GAT(**kwargs), 'l1'
    if model_name == 'GIN':
        return GIN(**kwargs), 'l1'
    if model_name == 'GraphSAGE':
        return GraphSAGE(**kwargs), 'l1'
    if model_name == 'MLP':
        return MLP(**kwargs), 'l1'
    raise ValueError(f'Invalid model name: {model_name}')

def trainer_factory(model_name: str, **kwargs):
    if model_name == 'NodeEdgeGNN':
        return DualRegressionTrainer(mode='node', **kwargs)
    return NodeRegressionTrainer(**kwargs)

def main():
    args = parse_args()
    logger = Logger(log_path=args.log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        with open(args.config_path) as f:
            config = yaml.safe_load(f)

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        data_file_path = config['dataset_parameters']['data_file_path']
        dataset_info = file_utils.read_shelve_file(data_file_path, 'dataset_info')
        logger.log(f'Using model: {args.model}')
        base_model_params = {
            'static_node_features': dataset_info['num_static_node_features'],
            'dynamic_node_features': dataset_info['num_dynamic_node_features'],
            'static_edge_features': dataset_info['num_static_edge_features'],
            'dynamic_edge_features': dataset_info['num_dynamic_edge_features'],
            'previous_timesteps': dataset_info['previous_timesteps'],
            'device': args.device,
        }
        model_params = config['model_parameters'][args.model]
        model, loss_func_key = model_factory(args.model, **model_params, **base_model_params)
        logger.log(f'Using loss function: {loss_func_key}')

        train_config = config['training_parameters']
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
        loss_func = get_loss_func(loss_func_key)
        trainer = trainer_factory(args.model, train_dataset=train_dataset, val_dataset=test_dataset, model=model,
                                        loss_func=loss_func, optimizer=optimizer, num_epochs=train_config['num_epochs'],
                                        device=args.device, logger=logger)
        trainer.train()
        trainer.validate()
        stats = trainer.get_stats()
        stats.print_stats_summary()
        # stats.plot_train_loss()

        logger.log('================================================')

    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
