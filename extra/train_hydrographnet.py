import sys
sys.path.append('..')

import numpy as np
import traceback
import torch
import os
import yaml

from argparse import ArgumentParser, Namespace
from datetime import datetime
from torch_geometric.loader import DataLoader
from train import get_loss_func_param, model_factory, trainer_factory
from utils import Logger, file_utils

from hydrographnet_flood_event_dataset import HydroGraphNetFloodEventDataset

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config_path", type=str, default='../configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='GCN', help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--model_dir", type=str, default='../saved_models', help='Path to directory to save trained models')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def main():
    # Constants
    batch_size = 1
    data_dir = ""
    n_time_steps = 2
    train_ids_file = "0_train.txt"
    num_input_features = 16
    num_output_features = 1

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

        dataset = HydroGraphNetFloodEventDataset(data_dir=data_dir,
                                                 prefix="M80",
                                                 n_time_steps=n_time_steps,
                                                 k=4,
                                                 hydrograph_ids_file=train_ids_file,
                                                 split="train",
                                                 logger=logger)
        logger.log(f'Loaded dataset with {len(dataset)} graphs.')

        # Training
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_NoPassing'] else args.model
        model_params = config['model_parameters'][model_key]
        logger.log(f'Using model: {args.model}')

        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Loss function
        loss_func_config = config['loss_func_parameters'][model_key] if model_key in config['loss_func_parameters'] else {}
        train_config = config['training_parameters']

        model = model_factory(args.model,
                              input_features=num_input_features,
                              output_features=num_output_features,
                              device=args.device,
                              **model_params)
        if args.debug:
            num_train_params = model.get_model_size()
            logger.log(f'Number of trainable model parameters: {num_train_params}')

        loss_func = get_loss_func_param(args.model, **loss_func_config)
        loss_func_name = loss_func.__name__ if hasattr(loss_func, '__name__') else loss_func.__class__.__name__
        logger.log(f"Using loss function: {loss_func_name}")

        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

        train_datasets = [DataLoader(dataset, batch_size=batch_size)]
        trainer = trainer_factory(args.model, train_datasets=train_datasets, model=model,
                                        loss_func=loss_func, optimizer=optimizer, num_epochs=train_config['num_epochs'],
                                        device=args.device, debug=args.debug, logger=logger)
        trainer.train()
        trainer.print_stats_summary()

        if args.model_dir is not None:
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)

            model_name = f'{args.model}_for_HydroGraphNet_{curr_date_str}.pt'
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
