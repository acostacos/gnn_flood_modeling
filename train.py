import numpy as np
import os
import traceback
import torch
import yaml

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import InMemoryFloodEventDataset
from models import GAT, GCN, GraphSAGE, GIN, GNNNoPassing, MLP, NodeEdgeGNN, NodeEdgeGNNNoPassing
from training import NodeRegressionTrainer, DualRegressionTrainer
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected, Compose
from utils import Logger, file_utils, loss_func_utils

torch.serialization.add_safe_globals([datetime])

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config_path", type=str, default='configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='NodeEdgeGNN', help='Model to use for training')
    parser.add_argument('--test_datasets', nargs='+', type=str, default=None, help='List of datasets to test with. The rest will only be used for training.')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--model_dir", type=str, default='saved_models', help='Path to directory to save trained models')
    parser.add_argument("--stats_dir", type=str, default=None, help='Path to directory to save training stats')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def model_factory(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name == 'NodeEdgeGNN':
        return NodeEdgeGNN(**kwargs)
    if model_name == 'NodeEdgeGNN_NoPassing':
        return NodeEdgeGNNNoPassing(**kwargs)
    if model_name == 'GCN':
        return GCN(**kwargs)
    if model_name == 'GAT':
        return GAT(**kwargs)
    if model_name == 'GIN':
        return GIN(**kwargs)
    if model_name == 'GraphSAGE':
        return GraphSAGE(**kwargs)
    if model_name == 'GNNNoPassing':
        return GNNNoPassing(**kwargs)
    if model_name == 'MLP':
        return MLP(**kwargs)
    raise ValueError(f'Invalid model name: {model_name}')

def trainer_factory(model_name: str, **kwargs):
    if 'NodeEdgeGNN' in model_name:
        return DualRegressionTrainer(**kwargs)
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

        # TODO: Implement FloodEventDataset for disk storage mode
        # dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        dataset_class = InMemoryFloodEventDataset
        transform = Compose([ToUndirected()])

        datasets = {}
        for event_key, event_parameters in dataset_parameters['flood_events'].items():
            dataset = dataset_class(**event_parameters,
                        dataset_info_path=dataset_info_path,
                        feature_stats_file=dataset_parameters['feature_stats_file'],
                        previous_timesteps=dataset_parameters['previous_timesteps'],
                        node_feat_config=dataset_parameters['node_features'],
                        edge_feat_config=dataset_parameters['edge_features'],
                        normalize=dataset_parameters['normalize'],
                        trim_from_peak_water_depth=dataset_parameters['trim_from_peak_water_depth'],
                        logger=logger,
                        transform=transform,
                        debug=args.debug)
            datasets[event_key] = DataLoader(dataset, batch_size=train_config['batch_size'])
        dataset_info = file_utils.read_yaml_file(dataset_info_path)

        # Training
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_NoPassing'] else args.model
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

        test_dataset_keys = args.test_datasets if args.test_datasets is not None else list(datasets.keys())
        for event_key in test_dataset_keys:
            if event_key not in datasets:
                raise ValueError(f'Test dataset {event_key} not found in datasets. Check your config file.')

        # Loss function
        loss_func_key = model_params.pop('loss_func', None)
        assert loss_func_key is not None, 'Loss function key not found in model parameters.'
        loss_func_config = model_params.pop('loss_func_parameters') if 'loss_func_parameters' in model_params else {}
        loss_func = loss_func_utils.get_loss_func(loss_func_key, **loss_func_config)

        loss_func_name = loss_func.__name__ if hasattr(loss_func, '__name__') else loss_func.__class__.__name__
        logger.log(f"Using loss function: {loss_func_name}")

        for i, event_key in enumerate(test_dataset_keys):
            model = model_factory(args.model, **model_params, **base_model_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

            if i == 0:
                if args.debug:
                    num_train_params = model.get_model_size()
                    logger.log(f'Number of trainable model parameters: {num_train_params}')

            train_datasets = [d for k, d in datasets.items() if k != event_key]
            logger.log(f"Training with {', '.join([k for k in datasets.keys() if k != event_key])}. For testing on {event_key}.")

            trainer = trainer_factory(args.model, train_datasets=train_datasets, model=model,
                                            loss_func=loss_func, optimizer=optimizer, num_epochs=train_config['num_epochs'],
                                            device=args.device, debug=args.debug, logger=logger)
            trainer.train()
            trainer.print_stats_summary()
            if args.stats_dir is not None:
                saved_metrics_path = os.path.join(args.stats_dir, f'{args.model}_{event_key}_train_metrics.npz')
                trainer.save_training_stats(saved_metrics_path)

            if args.model_dir is not None:
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)

                model_name = f'{args.model}_for_{event_key}_{curr_date_str}.pt'
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
