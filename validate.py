import numpy as np
import os
import traceback
import time
import torch
import re
import yaml

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import InMemoryFloodEventDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected, Compose
from train import model_factory
from utils import Logger, file_utils, metric_utils

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

        model.eval()
        with torch.no_grad():
            # Features per timestep
            pred_list = []
            target_list = []

            # Metrics per timestep
            rmse_list = []
            mae_list = []
            nse_list = []
            csi_list = []
            rmse_flooded_list = []
            mae_flooded_list = []
            nse_flooded_list = []

            sliding_window_length = 1 * (previous_timesteps+1) # Water Level at the end of node features
            wl_sliding_window = dataset[0].x.clone()[:, -sliding_window_length:]
            wl_sliding_window = wl_sliding_window.to(args.device)
            start_time = time.time()

            for graph in data_loader:
                graph = graph.to(args.device)
                graph.x = torch.concat((graph.x[:, :-sliding_window_length], wl_sliding_window), dim=1)

                pred = model(graph)
                wl_sliding_window = torch.concat((wl_sliding_window[:, 1:], pred), dim=1)

                label = graph.y

                pred = pred.cpu()
                label = label.cpu()

                # Get elevation from the graph data
                ELEVATION_IDX = 2
                elevation = graph.x[:, ELEVATION_IDX][:, None].cpu()
                denormalized_elevation = dataset._denormalize_features('elevation', elevation)

                # Convert water level to water depth
                denormalized_pred = dataset._denormalize_features('water_level', pred)
                denormalized_label = dataset._denormalize_features('water_level', label)
                water_depth_pred = denormalized_pred - denormalized_elevation
                water_depth_label = denormalized_label - denormalized_elevation

                pred_list.append(water_depth_pred)
                target_list.append(water_depth_label)

                rmse = metric_utils.RMSE(water_depth_pred, water_depth_label)
                rmse_list.append(rmse)
                mae = metric_utils.MAE(water_depth_pred, water_depth_label)
                mae_list.append(mae)
                nse = metric_utils.NSE(water_depth_pred, water_depth_label)
                nse_list.append(nse)

                csi = metric_utils.CSI(water_depth_pred, water_depth_label, threshold=0.05)
                csi_list.append(csi)

                # Compute metrics for flooded areas only
                binary_pred = metric_utils.convert_water_depth_to_binary(water_depth_pred, water_threshold=0.05)
                binary_label = metric_utils.convert_water_depth_to_binary(water_depth_label, water_threshold=0.05)
                flooded_mask = binary_pred | binary_label
                flooded_pred = water_depth_pred[flooded_mask]
                flooded_label = water_depth_label[flooded_mask]

                rmse_flooded = metric_utils.RMSE(flooded_pred, flooded_label)
                rmse_flooded_list.append(rmse_flooded)
                mae_flooded = metric_utils.MAE(flooded_pred, flooded_label)
                mae_flooded_list.append(mae_flooded)
                nse_flooded = metric_utils.NSE(flooded_pred, flooded_label)
                nse_flooded_list.append(nse_flooded)

        end_time = time.time()
        rmse_np = np.array(rmse_list)
        mae_np = np.array(mae_list)
        nse_np = np.array(nse_list)
        csi_np = np.array(csi_list)
        rmse_flooded_np = np.array(rmse_flooded_list)
        mae_flooded_np = np.array(mae_flooded_list)
        nse_flooded_np = np.array(nse_flooded_list)

        logger.log(f'Inference time for one timestep: {(end_time - start_time)/len(pred_list):.4f} seconds')
        logger.log(f'Average RMSE: {rmse_np.mean():.4f}')
        logger.log(f'Average RMSE (flooded): {rmse_flooded_np.mean():.4f}')
        logger.log(f'Average MAE: {mae_np.mean():.4f}')
        logger.log(f'Average MAE (flooded): {mae_flooded_np.mean():.4f}')
        logger.log(f'Average NSE: {nse_np.mean():.4f}')
        logger.log(f'Average NSE (flooded): {nse_flooded_np.mean():.4f}')
        logger.log(f'Average CSI: {csi_np.mean():.4f}')

        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            saved_metrics_name = f'{args.model}_{event_key}_metrics.npz'
            saved_metrics_path = os.path.join(args.output_dir, saved_metrics_name)
            np.savez(saved_metrics_path,
                     pred=np.array(pred_list), target=np.array(target_list),
                     rmse=rmse_np, mae=mae_np, nse=nse_np, csi=csi_np,
                     rmse_flooded=rmse_flooded_np, mae_flooded=mae_flooded_np, nse_flooded=nse_flooded_np)

            logger.log(f'Saved metrics to: {saved_metrics_path}')

        logger.log('================================================')

    except yaml.YAMLError as e:
        logger.log(f'Error loading config YAML file. Error: {e}')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
