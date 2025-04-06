import numpy as np
import os
import traceback
import torch
import yaml

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import FloodEventDataset, InMemoryFloodEventDataset
from models import GAT, GCN, GraphSAGE, GIN, MLP, NodeEdgeGNN, NodeEdgeGNNNoPassing, SWEGNN
from training import NodeRegressionTrainer, DualRegressionTrainer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, ToUndirected
from typing import Tuple
from utils import Logger, file_utils
from utils.loss_func_utils import get_loss_func

torch.serialization.add_safe_globals([datetime])

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='NodeEdgeGNN_Dual', help='Model to use for training')
    parser.add_argument('--test_datasets', nargs='+', type=str, default=['all'], help='List of datasets to test with. The rest will only be used for testing.')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--model_dir", type=str, default=None, help='Path to directory to save trained models')
    return parser.parse_args()

def model_factory(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name == 'NodeEdgeGNN' or model_name == 'NodeEdgeGNN_Dual':
        return NodeEdgeGNN(**kwargs)
    if model_name == 'NodeEdgeGNN_NoPassing':
        return NodeEdgeGNNNoPassing(**kwargs)
    if model_name == 'SWEGNN':
        return SWEGNN(**kwargs)
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    if model_name == 'GIN':
        return GIN(**kwargs)
    if model_name == 'GraphSAGE':
        return GraphSAGE(**kwargs)
    if model_name == 'MLP':
        return MLP(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

def get_loss_func_params(model_name: str, **kwargs) -> str | Tuple[str, str]:
    if model_name == 'NodeEdgeGNN_Dual':
        return {
            'loss_func': get_loss_func(loss_func_name='scaled_l1', scale=kwargs['water_level_weight']),
            'edge_loss_func': get_loss_func(loss_func_name='scaled_l1', scale=kwargs['velocity_weight']),
        }
    return {'loss_func': get_loss_func('l1')}

def trainer_factory(model_name: str, **kwargs):
    if model_name == 'NodeEdgeGNN_Dual':
        return DualRegressionTrainer(mode='dual', **kwargs)
    if model_name == 'NodeEdgeGNN' or model_name == 'NodeEdgeGNN_NoPassing':
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

        config = file_utils.read_yaml_file(args.config_path)

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Load datasets
        dataset_parameters = config['dataset_parameters']
        dataset_info_path = dataset_parameters['dataset_info_path']
        storage_mode = dataset_parameters['storage_mode']
        train_config = config['training_parameters']

        dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        data_loader_params = {
            'batch_size': train_config['batch_size'],
            # 'pin_memory': True,
            # 'num_workers': 2,
            # 'persistent_workers': True,
        }
        transform = Compose([ToUndirected()])

        datasets = {}
        for event_key, event_parameters in dataset_parameters['flood_events'].items():
            dataset = dataset_class(**event_parameters,
                        dataset_info_path=dataset_info_path,
                        previous_timesteps=dataset_parameters['previous_timesteps'],
                        node_features=dataset_parameters['node_features'],
                        edge_features=dataset_parameters['edge_features'],
                        transform=transform,
                        debug=args.debug)
            datasets[event_key] = DataLoader(dataset, **data_loader_params)
        dataset_info = file_utils.read_yaml_file(dataset_info_path)

        # Training
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_Dual', 'NodeEdgeGNN_NoPassing'] else args.model
        model_params = config['model_parameters'][model_key]
        logger.log(f'Using model: {args.model}')

        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        base_model_params = {
            'static_node_features': dataset_info['num_static_node_features'],
            'dynamic_node_features': dataset_info['num_dynamic_node_features'],
            'static_edge_features': dataset_info['num_static_edge_features'],
            'dynamic_edge_features': dataset_info['num_dynamic_edge_features'],
            'previous_timesteps': dataset_info['previous_timesteps'],
            'device': args.device,
        }

        test_dataset_keys = args.test_datasets if args.test_datasets != ['all'] else list(datasets.keys())
        for event_key in test_dataset_keys:
            if event_key not in datasets:
                raise ValueError(f'Test dataset {event_key} not found in datasets. Check your config file.')

        # Loss function
        loss_func_config = config['loss_func_parameters'][model_key] if model_key in config['loss_func_parameters'] else {}
        loss_func_params = get_loss_func_params(args.model, **loss_func_config)
        loss_func_names = [(lf.__name__ if hasattr(lf, '__name__') else type(lf).__name__) for lf in loss_func_params.values()]
        logger.log(f"Using loss function: {', '.join(loss_func_names)}")

        for event_key in test_dataset_keys:
            train_datasets = [d for k, d in datasets.items() if k != event_key]
            test_dataset = datasets[event_key]
            logger.log(f"Training with {', '.join([k for k in datasets.keys() if k != event_key])}. Testing on {event_key}.")
            model = model_factory(args.model, **model_params, **base_model_params)

            if args.debug:
                num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.log(f'Number of trainable model parameters: {num_train_params}')

            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
            trainer = trainer_factory(args.model, train_datasets=train_datasets, val_dataset=test_dataset, model=model,
                                            optimizer=optimizer, num_epochs=train_config['num_epochs'],
                                            device=args.device, debug=args.debug, logger=logger, **loss_func_params)
            trainer.train()
            trainer.validate()
            stats = trainer.get_stats()
            stats.print_stats_summary()
            # stats.plot_train_loss()

            if args.model_dir is not None:
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                model_name = f'{args.model}_{event_key}_{curr_date_str}.pt'
                model_path = os.path.join(args.model_dir, model_name)
                torch.save(model.state_dict(), model_path)

                logger.log(f'Saved model to: {model_path}')

        logger.log('================================================')

    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
