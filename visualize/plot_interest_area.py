import sys
sys.path.append('..')

import torch

from train import model_factory
from data import FloodEventDataset, InMemoryFloodEventDataset
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
    dataset = dataset_class(**event_parameters,
                dataset_info_path=dataset_info_path,
                previous_timesteps=dataset_parameters['previous_timesteps'],
                node_features=dataset_parameters['node_features'],
                edge_features=dataset_parameters['edge_features'])
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

if __name__ == "__main__":
    main()
