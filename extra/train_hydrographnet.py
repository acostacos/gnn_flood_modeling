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
from train import model_factory, get_loss_func_w_param
from training.training_stats import TrainingStats
from utils import Logger, file_utils

from hydrographnet_flood_event_dataset import HydroGraphNetFloodEventDataset
from hydrographnet_utils import compute_physics_loss

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config_path", type=str, default='../configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='GCN', help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--model_dir", type=str, default='../saved_models', help='Path to directory to save trained models')
    parser.add_argument("--stats_dir", type=str, default=None, help='Path to directory to save training stats')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def main():
    # Constants
    batch_size = 1
    data_dir = ""
    n_time_steps = 2
    train_ids_file = "0_train.txt"
    use_physics_loss = True
    num_input_features = 16
    num_output_features = 2 if use_physics_loss else 1 # Water depth and volume if using physics loss, water depth only otherwise
    num_epochs = 1

    if use_physics_loss:
        assert batch_size == 1, 'Batch size must be 1 when using physics loss.'

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
                                                 return_physics=use_physics_loss,
                                                 logger=logger)
        logger.log(f'Loaded dataset with {len(dataset)} total timesteps.')
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # Training
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_NoPassing'] else args.model
        model_params = config['model_parameters'][model_key]
        logger.log(f'Using model: {args.model}')

        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Loss function
        train_config = config['training_parameters']
        loss_func_key = model_params.pop('loss_func', None)
        assert loss_func_key is not None, 'Loss function key not found in model parameters.'
        loss_func_config = model_params.pop('loss_func_parameters') if 'loss_func_parameters' in model_params else {}
        loss_func = get_loss_func_w_param(args.model, **loss_func_config)
        if args.debug:
            loss_func_name = loss_func.__name__ if hasattr(loss_func, '__name__') else loss_func.__class__.__name__
            logger.log(f"Using loss function: {loss_func_name}")

        model = model_factory(args.model,
                              input_features=num_input_features,
                              output_features=num_output_features,
                              device=args.device,
                              **model_params)
        if args.debug:
            num_train_params = model.get_model_size()
            logger.log(f'Number of trainable model parameters: {num_train_params}')

        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

        training_stats = TrainingStats(logger=logger)
        training_stats.start_train()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_phy_loss = 0.0
            num_batches = 0

            for batch, physics_data in data_loader:
                optimizer.zero_grad()

                batch = batch.to(args.device)
                pred = model(batch)

                label = batch.y
                loss = loss_func(pred, label)
                running_loss += loss.item()

                if use_physics_loss and len(physics_data) > 0:
                    phy_loss = compute_physics_loss(pred, physics_data, batch)
                    loss = loss + phy_loss # Physics loss weight = 1
                    running_phy_loss += phy_loss.item()

                loss.backward()
                optimizer.step()
                num_batches += 1

            epoch_pred_loss = running_loss / num_batches
            epoch_phy_loss = running_phy_loss / num_batches
            epoch_total_loss = epoch_pred_loss + epoch_phy_loss
            logger.log(f'Epoch [{epoch + 1}/{num_epochs}]:')
            logger.log(f'\tPred Loss: {epoch_pred_loss:.4f}')
            logger.log(f'\tPhysics Loss: {epoch_phy_loss:.4f}')
            logger.log(f'\tTotal Loss: {epoch_total_loss:.4f}')
            training_stats.add_train_loss(epoch_total_loss)

        additional_info = {
            'Final Pred Loss': epoch_pred_loss,
            'Final Physics Loss': epoch_total_loss,
        }
        training_stats.update_additional_info(additional_info)
        training_stats.end_train()
        training_stats.print_stats_summary()
        if args.stats_dir is not None:
            saved_metrics_path = os.path.join(args.stats_dir, f'{args.model}_HydroGraphNet_train_metrics.npz')
            training_stats.save_stats(saved_metrics_path)

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
