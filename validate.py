import numpy as np
import os
import traceback
import torch
import re
import yaml

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import InMemoryFloodEventDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected, Compose
from train import model_factory
from utils import Logger, file_utils
from validation import AutoregressionValidator

torch.serialization.add_safe_globals([datetime])

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config_path", type=str, default='configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='NodeEdgeGNN', help='Model to use for training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model file')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--output_dir", type=str, default='saved_metrics', help='Path to directory to save metrics')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def parse_model_path(model_path: str, model_name: str) -> str:
    assert model_path is not None, 'Model path must be provided.'
    if not os.path.exists(model_path):
        raise ValueError(f'Model path {model_path} does not exist.')

    filename = os.path.basename(model_path)
    filename_regex = r'^([a-zA-Z0-9_]+)_for_([a-zA-Z0-9]+)_[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}.pt$'
    match = re.match(filename_regex, filename)
    if not match:
        raise ValueError(f'Model name {filename} does not match expected format.')
    
    if match.group(1) != model_name:
        raise ValueError(f'Model name {match.group(1)} does not match expected model name {model_name}.')

    event_key = match.group(2)
    return event_key


def main():
    args = parse_args()
    logger = Logger(log_path=args.log_path)

    try:
        logger.log('================================================')

        event_key = parse_model_path(args.model_path, args.model)
        logger.log(f'Loading {args.model} model from {args.model_path}')

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        config = file_utils.read_yaml_file(args.config_path)

        # Load validation dataset
        dataset_parameters = config['dataset_parameters']
        flood_events = dataset_parameters['flood_events']
        dataset_info_path = dataset_parameters['dataset_info_path']
        storage_mode = dataset_parameters['storage_mode']

        if event_key not in flood_events:
            raise ValueError(f'Event key {event_key} not found in dataset parameters.')

        event_parameters = flood_events[event_key]
        # TODO: Implement FloodEventDataset for disk storage mode
        # dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
        dataset_class = InMemoryFloodEventDataset
        transform = Compose([ToUndirected()])
        dataset = dataset_class(**event_parameters,
                    dataset_info_path=dataset_info_path,
                    feature_stats_file=dataset_parameters['feature_stats_file'],
                    previous_timesteps=dataset_parameters['previous_timesteps'],
                    node_feat_config=dataset_parameters['node_features'],
                    edge_feat_config=dataset_parameters['edge_features'],
                    normalize=dataset_parameters['normalize'],
                    logger=logger,
                    transform=transform,
                    debug=args.debug)
        data_loader = DataLoader(dataset, batch_size=1)
        dataset_info = file_utils.read_yaml_file(dataset_info_path)

        # Load model
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_NoPassing'] else args.model
        model_params = config['model_parameters'][model_key]

        previous_timesteps = dataset_info['previous_timesteps']
        static_node_features = dataset_info['num_static_node_features']
        dynamic_node_features = dataset_info['num_dynamic_node_features']
        dynamic_edge_features = dataset_info['num_dynamic_edge_features']
        base_model_params = {
            'static_node_features': static_node_features,
            'dynamic_node_features': dynamic_node_features,
            'static_edge_features': dataset_info['num_static_edge_features'],
            'dynamic_edge_features': dynamic_edge_features,
            'previous_timesteps': previous_timesteps,
            'device': args.device,
        }

        model = model_factory(args.model, **model_params, **base_model_params)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

        validator = AutoregressionValidator(val_dataset=data_loader,
                                            model=model,
                                            device=args.device,
                                            denormalize_func=dataset._denormalize_features,
                                            debug=args.debug,
                                            logger=logger)
        saved_metrics_path = os.path.join(args.output_dir, f'{args.model}_{event_key}_metrics.npz') if args.output_dir is not None else None
        validator.validate(save_stats_path=saved_metrics_path)

        logger.log('================================================')

    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
