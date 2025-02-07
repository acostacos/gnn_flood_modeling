import geopandas as gpd
import h5py
import json
import numpy as np

from typing import Any

def read_json_file(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def read_shp_file_as_numpy(filepath: str, columns: str | list) -> np.ndarray:
    file = gpd.read_file(filepath)
    np_data = file[columns].to_numpy()
    return np_data

def read_hdf_file_as_numpy(filepath: str, property_path: str, separator: str = '.') -> np.ndarray:
    with h5py.File(filepath, 'r') as hec:
        data = get_property_from_path(hec, property_path, separator)
        np_data = np.array(data)
    return np_data

def get_property_from_path(dict: dict, dict_path: str, separator: str = '.') -> Any:
    keys = dict_path.split(sep=separator)
    d = dict
    for key in keys:
        if key in d:
            d = d[key]
    return d
