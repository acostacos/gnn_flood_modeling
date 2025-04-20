import sys
sys.path.append('..')

import numpy as np
import torch

from train import model_factory
from data import FloodEventDataset, InMemoryFloodEventDataset
from utils import file_utils
from utils.loss_func_utils import get_loss_func

from plot_constants import WOLLOMBI_LR_INTEREST_AREA_NODES

MESH = 'lr'
EVENT_NAME = 'lrp01'
MODEL = 'GCN'
SAVED_MODEL_PATH = '../saved_models/gcn/GCN_lrp01_2025-04-17_16-02-34.pt'

def get_interest_area_mask():
    flattened = np.array(WOLLOMBI_LR_INTEREST_AREA_NODES).flatten().tolist()
    mask = list(set(flattened))
    return mask

def main():
    np.random.seed(42)
    torch.manual_seed(42)

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
                edge_features=dataset_parameters['edge_features'],
                normalize=dataset_parameters['normalize'],)
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
    loss_func = get_loss_func('l1') # For now use L1 loss

    model.eval()

    total_running_loss = 0.0
    aoi_running_loss = 0.0
    aoi_mask = get_interest_area_mask()

    len_dataset = len(dataset)
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to(device)
            output = model(batch)

            if MODEL == 'NodeEdgeGNN':
                output = output[0]

            label = batch.y

            total_running_loss += loss_func(output, label).item()
            aoi_running_loss += loss_func(output[aoi_mask], label[aoi_mask]).item()

    total_avg_loss = total_running_loss  / len_dataset
    aoi_avg_loss = aoi_running_loss / len_dataset
    print(f'Total Validtion Loss (Interest Area): {aoi_avg_loss:.4f}')
    print(f'Total Validation Loss: {total_avg_loss:.4f}')
    percentage_loss = (aoi_avg_loss / total_avg_loss) * 100
    print(f'Percentage of Interest Area Loss: {percentage_loss:.2f}%')

if __name__ == "__main__":
    main()
