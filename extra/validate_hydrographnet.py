import sys
sys.path.append('..')

import numpy as np
import traceback
import torch
import os
import yaml

from argparse import ArgumentParser, Namespace
from train import model_factory
from utils import Logger, file_utils
from validation.validation_stats import ValidationStats

from hydrographnet_flood_event_dataset import HydroGraphNetFloodEventDataset

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config_path", type=str, default='../configs/config.yaml', help='Path to training config file')
    parser.add_argument("--model", type=str, default='GCN', help='Model to use for validation')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model file')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--log_path", type=str, default=None, help='Path to log file')
    parser.add_argument("--output_dir", type=str, default='../saved_metrics', help='Path to directory to save metrics')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def main():
    # Constants
    data_dir = ""
    n_time_steps = 2
    test_ids_file = "0_test.txt"
    num_input_features = 16
    num_output_features = 1
    rollout_length = 30

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
                                                 hydrograph_ids_file=test_ids_file,
                                                 split="test",
                                                 rollout_length=rollout_length,
                                                 logger=logger)
        logger.log(f'Loaded dataset with {len(dataset)} total timesteps.')

        # Load model
        model_key = 'NodeEdgeGNN' if args.model in ['NodeEdgeGNN_NoPassing'] else args.model
        model_params = config['model_parameters'][model_key]

        model = model_factory(args.model,
                              input_features=num_input_features,
                              output_features=num_output_features,
                              device=args.device,
                              **model_params)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))

        # Loop over each test hydrograph.
        WATER_DEPTH_IDX = 12
        all_rmse_all = []
        global_idx = 0

        model.eval()
        with torch.no_grad():
            while global_idx < len(dataset):
                rmse_list = [] # RMSE at each rollout step.
                validation_stats = ValidationStats(logger=logger)

                wd_sliding_window = dataset[global_idx].x.clone()[:, WATER_DEPTH_IDX:(WATER_DEPTH_IDX+n_time_steps)]
                wd_sliding_window = wd_sliding_window.to(args.device)
                validation_stats.start_validate()

                for idx in range(rollout_length):
                    graph = dataset[global_idx]
                    graph = graph.to(args.device)
                    graph.x[:, WATER_DEPTH_IDX:(WATER_DEPTH_IDX+n_time_steps)] = wd_sliding_window

                    pred = model(graph)

                    wd_sliding_window = torch.concat((wd_sliding_window[:, 1:], pred), dim=1)

                    label = graph.y

                    # Clip negative values for water depth
                    wd_mean = dataset.dynamic_stats["water_depth"]["mean"]
                    wd_std = dataset.dynamic_stats["water_depth"]["std"]
                    pred = dataset.denormalize(pred, wd_mean, wd_std)
                    pred = torch.clip(pred, min=0)
                    label = dataset.denormalize(label, wd_mean, wd_std)
                    label = torch.clip(label, min=0)

                    validation_stats.update_stats_for_epoch(pred.cpu(),
                                                    label.cpu(),
                                                    water_threshold=0.05)

                    rmse = torch.sqrt(torch.mean((pred - label) ** 2)).item()
                    rmse_list.append(rmse)

                    global_idx += 1

                validation_stats.end_validate()

                all_rmse_all.append(rmse_list)
                mean_rmse_sample = sum(rmse_list) / len(rmse_list)
                dyn_data_idx = (global_idx // rollout_length) - 1
                sample_id = dataset.dynamic_data[dyn_data_idx].get('hydro_id', idx)
                print(f"Hydrograph {sample_id}: Mean RMSE = {mean_rmse_sample:.4f}")

                validation_stats.print_stats_summary()

                save_stats_path = os.path.join(args.output_dir, f'{args.model}_{sample_id}_metrics.npz') if args.output_dir is not None else None
                if save_stats_path is not None:
                    validation_stats.save_stats(save_stats_path)

        # HydroGraphNet validation stats.
        all_rmse_tensor = torch.tensor(all_rmse_all)
        overall_mean_rmse = torch.mean(all_rmse_tensor, dim=0)
        overall_std_rmse = torch.std(all_rmse_tensor, dim=0)
        print("Overall Mean RMSE over rollout steps:", overall_mean_rmse)
        print("Overall Std RMSE over rollout steps:", overall_std_rmse)

        logger.log('================================================')

    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
