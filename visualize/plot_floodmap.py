import sys
sys.path.append('..')

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import torch

from train import model_factory
from data import FloodEventDataset, InMemoryFloodEventDataset
from torch_geometric.transforms import Compose, ToUndirected
from utils import file_utils

TIMESTEP_IDX = 152
NORMALIZE_WITH_ENTIRE_DATASET = False
MESH = 'lr'
EVENT_NAME = 'lrp08'
HEC_RAS_FILENAME = 'M01.p08.hdf'
CELL_SHP_FILENAME = 'cell_centers.shp'
LINK_SHP_FILENAME = 'links.shp'
MODEL = 'NodeEdgeGNN_Dual'
SAVED_MODEL_PATH = '../saved_models/nodeedge_dual/NodeEdgeGNN_Dual_lrp01_2025-03-30_21-22-59.pt'


def plot_node_flood_map(node_pred, node_ground_truth, timestep):
    # Get range of values for water level
    if NORMALIZE_WITH_ENTIRE_DATASET:
        feature_metadata_path = f"../data/feature_metadata.yaml"
        feature_metadata = file_utils.read_yaml_file(feature_metadata_path)
        hec_ras_path = f'../data/datasets/{MESH}/{EVENT_NAME}/raw/{HEC_RAS_FILENAME}'

        water_level_metadata = feature_metadata['node_features']['water_level']
        water_level = file_utils.read_hdf_file_as_numpy(filepath=hec_ras_path, property_path=water_level_metadata['path'])
        min_water_level, max_water_level = np.min(water_level), np.max(water_level)
        print(f'Water Level: min = {min_water_level}, max = {max_water_level}')
        norm = plt.Normalize(vmin=min_water_level, vmax=max_water_level)
    else:
        node_pred_min, node_pred_max = node_pred.min(), node_pred.max()
        node_gt_min, node_gt_max = node_ground_truth.min(), node_ground_truth.max()
        print(f'Node Prediction: min = {node_pred_min}, max = {node_pred_max}')
        print(f'Node Ground Truth: min = {node_gt_min}, max = {node_gt_max}')
        norm = plt.Normalize(vmin=min(node_pred_min, node_gt_min), vmax=max(node_pred_max, node_gt_max))

    cell_shp_path = f'../data/datasets/{MESH}/{EVENT_NAME}/raw/{CELL_SHP_FILENAME}'
    node_df = gpd.read_file(cell_shp_path)
    value_column = 'water_level'

    fig, ax = plt.subplots(figsize=(10,3), ncols=3)

    cmap = plt.get_cmap('RdYlGn_r') 
    shared_plot_kwargs = {
        'cmap': cmap,
        'norm': norm,
        'column': value_column,
        'linewidth': 0.3,
        'markersize': 0.5,
        'legend': True,
        'legend_kwds': {'label': "Water Level", 'orientation': "vertical"},
    }

    # Prediction
    np_node_pred = node_pred.cpu().numpy()
    pred_node_df = node_df.copy()
    pred_node_df[value_column] = np_node_pred

    pred_node_df.plot(ax=ax[0], **shared_plot_kwargs)
    ax[0].set_title('Node Prediction')
    ax[0].set_axis_off()

    # Ground Truth
    np_node_ground_truth = node_ground_truth.cpu().numpy()
    gt_node_df = node_df.copy()
    gt_node_df[value_column] = np_node_ground_truth
    gt_node_df.plot(ax=ax[1], **shared_plot_kwargs)
    ax[1].set_title('Node Ground Truth')
    ax[1].set_axis_off()

    # Difference
    node_diff = np.abs(np_node_pred - np_node_ground_truth)
    print(f'Node Difference: min = {node_diff.min()}, max = {node_diff.max()}')
    diff_node_df = node_df.copy()
    diff_node_df[value_column] = node_diff
    diff_node_df.plot(ax=ax[2],
                      cmap='Blues',
                      column=value_column,
                      linewidth=0.3,
                      markersize=0.5,
                      legend=True,
                      legend_kwds={'label': "Difference", 'orientation': "vertical"})
    ax[2].set_title('Node Difference')
    ax[2].set_axis_off()

    fig.suptitle(f"Water Level Heatmap for Mesh Nodes at timestep {timestep}", fontsize=16)
    fig.tight_layout()
    fig.savefig("plots/lr_node_flood_map.jpg", format='jpg', dpi=300, bbox_inches='tight')


def plot_edge_flood_map(edge_pred, edge_ground_truth, timestep):
    # Get range of values for velocity
    if NORMALIZE_WITH_ENTIRE_DATASET:
        feature_metadata_path = f"../data/feature_metadata.yaml"
        feature_metadata = file_utils.read_yaml_file(feature_metadata_path)
        hec_ras_path = f'../data/datasets/{MESH}/{EVENT_NAME}/raw/{HEC_RAS_FILENAME}'

        velocity_metadata = feature_metadata['edge_features']['velocity']
        velocity = file_utils.read_hdf_file_as_numpy(filepath=hec_ras_path, property_path=velocity_metadata['path'])
        min_velocity, max_velocity = np.min(velocity), np.max(velocity)
        print(f'Velocity: min = {min_velocity}, max = {max_velocity}')
        norm = plt.Normalize(vmin=min_velocity, vmax=max_velocity)
    else:
        edge_pred_min, edge_pred_max = edge_pred.min(), edge_ground_truth.max()
        edge_gt_min, edge_gt_max = edge_ground_truth.min(), edge_ground_truth.max()
        print(f'Edge Prediction: min = {edge_pred_min}, max = {edge_pred_max}')
        print(f'Edge Ground Truth: min = {edge_gt_min}, max = {edge_gt_max}')
        norm = plt.Normalize(vmin=min(edge_pred_min, edge_gt_min), vmax=max(edge_pred_max, edge_gt_max))

    # TODO: See if this is correct
    edge_pred = edge_pred[:edge_pred.shape[0] // 2]
    edge_ground_truth = edge_ground_truth[:edge_ground_truth.shape[0] // 2]

    link_shp_path = f'../data/datasets/{MESH}/{EVENT_NAME}/raw/{LINK_SHP_FILENAME}'
    edge_df = gpd.read_file(link_shp_path)
    # raw_edge_df = gpd.read_file(link_shp_path)
    # edge_df = pd.concat([raw_edge_df, raw_edge_df], axis=0)
    value_column = 'velocity'

    fig, ax = plt.subplots(figsize=(10,3), ncols=3)

    cmap = plt.get_cmap('Reds') 
    shared_plot_kwargs = {
        'cmap': cmap,
        'norm': norm,
        'column': value_column,
        'linewidth': 0.5,
        'legend': True,
        'legend_kwds': {'label': "Velocity", 'orientation': "vertical"},
    }

    # Prediction
    np_edge_pred = edge_pred.cpu().numpy()
    pred_edge_df = edge_df.copy()
    pred_edge_df[value_column] = np_edge_pred

    pred_edge_df.plot(ax=ax[0], **shared_plot_kwargs)
    ax[0].set_title('Edge Prediction')
    ax[0].set_axis_off()

    # Ground Truth
    np_edge_ground_truth = edge_ground_truth.cpu().numpy()
    gt_edge_df = edge_df.copy()
    gt_edge_df[value_column] = np_edge_ground_truth
    gt_edge_df.plot(ax=ax[1], **shared_plot_kwargs)
    ax[1].set_title('Edge Ground Truth')
    ax[1].set_axis_off()

    # Difference
    edge_diff = np.abs(np_edge_pred - np_edge_ground_truth)
    print(f'Edge Difference: min = {edge_diff.min()}, max = {edge_diff.max()}')
    diff_edge_df = edge_df.copy()
    diff_edge_df[value_column] = edge_diff
    diff_edge_df.plot(ax=ax[2],
                      cmap='Blues',
                      column=value_column,
                      linewidth=0.5,
                      legend=True,
                      legend_kwds={'label': "Difference", 'orientation': "vertical"})
    ax[2].set_title('Edge Difference')
    ax[2].set_axis_off()

    fig.suptitle(f"Velocity Heatmap for Mesh Edges at timestep {timestep}", fontsize=16)
    fig.tight_layout()
    fig.savefig("plots/lr_edge_flood_map.jpg", format='jpg', dpi=300, bbox_inches='tight')


def main():
    config_path = f"../configs/{f'{MESH}_' if MESH != 'init' else ''}config.yaml"
    config = file_utils.read_yaml_file(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    dataset_parameters = config['dataset_parameters']
    dataset_info_path = f"../{dataset_parameters['dataset_info_path']}"
    storage_mode = dataset_parameters['storage_mode']
    event_parameters = dataset_parameters['flood_events'][EVENT_NAME]
    event_parameters['root_dir'] = f"../{event_parameters['root_dir']}"

    dataset_class = FloodEventDataset if storage_mode == 'disk' else InMemoryFloodEventDataset
    transform = Compose([ToUndirected()])
    dataset = dataset_class(**event_parameters,
                dataset_info_path=dataset_info_path,
                previous_timesteps=dataset_parameters['previous_timesteps'],
                node_features=dataset_parameters['node_features'],
                edge_features=dataset_parameters['edge_features'],
                transform=transform)
    dataset_info = file_utils.read_yaml_file(dataset_info_path)

    # Initialize model
    model_key = 'NodeEdgeGNN' if MODEL == 'NodeEdgeGNN_Dual' else MODEL
    model_params = config['model_parameters'][model_key]

    base_model_params = {
        'static_node_features': dataset_info['num_static_node_features'],
        'dynamic_node_features': dataset_info['num_dynamic_node_features'],
        'static_edge_features': dataset_info['num_static_edge_features'],
        'dynamic_edge_features': dataset_info['num_dynamic_edge_features'],
        'previous_timesteps': dataset_info['previous_timesteps'],
        'device': device,
    }

    model = model_factory(MODEL, **model_params, **base_model_params)
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, weights_only=True))

    # Perform inference for 1 timestep
    timestep_data = dataset[TIMESTEP_IDX]

    model.eval()
    with torch.no_grad():
        timestep_data = timestep_data.to(device)
        node_pred, edge_pred = model(timestep_data)
    
    node_ground_truth = timestep_data.y
    edge_ground_truth = timestep_data.y_edge
    timestep = timestep_data.timestep

    # For node prediction, subtract elevation
    feature_metadata_path = f"../data/feature_metadata.yaml"
    feature_metadata = file_utils.read_yaml_file(feature_metadata_path)
    shp_path = f'../data/datasets/{MESH}/{EVENT_NAME}/raw/{CELL_SHP_FILENAME}'
    elevation_metadata = feature_metadata['node_features']['elevation']
    elevation = file_utils.read_shp_file_as_numpy(filepath=shp_path, columns=elevation_metadata['column'])
    elevation = torch.Tensor(elevation)[:, None].to(device)
    node_pred = node_pred - elevation
    node_ground_truth = node_ground_truth - elevation

    plot_node_flood_map(node_pred, node_ground_truth, timestep)

    # For edge prediction, get the absolute value of the values
    edge_pred = torch.abs(edge_pred)
    edge_ground_truth = torch.abs(edge_ground_truth)

    plot_edge_flood_map(edge_pred, edge_ground_truth, timestep)

if __name__ == '__main__':
    main()
