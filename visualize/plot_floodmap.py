import geopandas as gpd
import matplotlib.pyplot as plt
import torch

from data import FloodEventDataset, InMemoryFloodEventDataset
from torch_geometric.transforms import Compose, ToUndirected
from models import GAT, GCN, GraphSAGE, GIN, MLP, NodeEdgeGNN, SWEGNN
from utils import file_utils

def model_factory(model_name: str, **kwargs) -> torch.nn.Module:
    if model_name == 'NodeEdgeGNN' or model_name == 'NodeEdgeGNN_Dual':
        return NodeEdgeGNN(**kwargs)
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

def main():
    MESH = 'lr'
    EVENT_NAME = 'lrp01'
    CELL_SHP_FILENAME = 'LR_Mesh_200m.shp'
    LINK_SHP_FILENAME = 'links.shp'
    MODEL = 'NodeEdgeGNN'
    SAVED_MODEL_PATH = ''
    NODE_PLOT_TITLE = 'Flood Mapping for Low Resolution Mesh Cells'
    NODE_PLOT_PATH = 'visualize/node_lr_node_plot.png'
    EDGE_PLOT_TITLE = 'Flood Mapping for Low Resolution Mesh Edges'
    EDGE_PLOT_PATH = 'visualize/node_lr_edge_plot.png'


    config_path = f"configs/{f'{MESH}_' if MESH != 'init' else ''}config.yaml"
    cell_shp_path = f'data/datasets/{MESH}/{EVENT_NAME}/{CELL_SHP_FILENAME}'
    link_shp_path = f'data/datasets/{MESH}/{EVENT_NAME}/{LINK_SHP_FILENAME}'

    config = file_utils.read_yaml_file(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load dataset
    dataset_parameters = config['dataset_parameters']
    dataset_info_path = dataset_parameters['dataset_info_path']
    storage_mode = dataset_parameters['storage_mode']
    event_parameters = dataset_parameters['flood_events'][EVENT_NAME]

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

    model = model_factory(SAVED_MODEL_PATH, **model_params, **base_model_params)
    model.load_state_dict(torch.load(MODEL, weights_only=True))

    # Perform inference for 1 timestep
    idx = len(dataset) - 100
    timestep_data = dataset[idx]

    model.eval()
    with torch.no_grad():
        timestep_data = timestep_data.to(device)
        pred = model(timestep_data)
    
    ground_truth = timestep_data.y

    # Get max and min values
    pred_max = pred.max(), pred_min = pred.min()
    gt_max = ground_truth.max(), gt_min = ground_truth.min()
    print(f'Pred: max = {pred_max}, min = {pred_min}')
    print(f'Ground Truth: max = {gt_max}, min = {gt_min}')

    # TODO: Get an area of the dataset from prediction

    # TODO: Adjust based on data
    breaks = [0, 10, 20, 30, 40, float('inf')]

    # Create a colormap
    cmap = plt.get_cmap('RdYlGn_r')  # Reversed Red-Yellow-Green
    # TODO: # Adjust based on your data
    norm = plt.Normalize(vmin=0, vmax=40)

    # Node Visualization
    node_df = gpd.read_file(cell_shp_path)
    value_column = 'your_value_column'

    # TODO: Filter cells with area

    # Color cells based on prediction and ground truth
    node_df.plot(column=value_column, cmap=cmap, norm=norm, 
         edgecolor='black', linewidth=0.3, legend=True,
         legend_kwds={'label': "Value Ranges", 'orientation': "horizontal"})

    plt.title(NODE_PLOT_TITLE)
    plt.show()
    plt.savefig(NODE_PLOT_PATH)

    # TODO: Edge visualization
    # Edge Visualization

if __name__ == '__main__':
    main()
