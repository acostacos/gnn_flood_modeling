import torch
import numpy as np

from utils import file_utils

def get_edge_index(filepath: str) -> torch.Tensor:
    columns = ['from_node', 'to_node']
    data = file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return torch.Tensor(data).type(torch.int32).transpose(1, 0)

def get_cell_elevation(filepath: str) -> np.ndarray:
    columns = 'Elevation1'
    return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)

def get_edge_length(filepath: str) -> np.ndarray:
    columns = 'length'
    return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)

def get_edge_slope(filepath: str) -> np.ndarray:
    columns = 'slope'
    return file_utils.read_shp_file_as_numpy(filepath=filepath, columns=columns)
