import numpy as np
import torch

from datetime import datetime

def select_index(data: np.ndarray, index: int = 0, axis: int = 0) -> np.ndarray:
    return np.squeeze(np.take(data, indices=[index], axis=axis))

def byte_to_timestamp(data: np.ndarray[np.bytes_]) -> np.ndarray:
    def format(x: np.bytes_) -> datetime:
        TIMESTAMP_FORMAT = '%d%b%Y %H:%M:%S'
        time_str = x.decode('UTF-8')
        time_stamp = datetime.strptime(time_str, TIMESTAMP_FORMAT)
        return time_stamp

    vec_format = np.vectorize(format)
    time_series = vec_format(data)
    return list(time_series)

def to_torch_tensor_w_transpose(data: np.ndarray) -> torch.Tensor:
    return torch.Tensor(data).type(torch.int64).transpose(1, 0)


TRANSFORM_MAP = {
    'direction_x': lambda x: select_index(x, index=0, axis=1),
    'direction_y': lambda x: select_index(x, index=1, axis=1),
    'face_length': lambda x: select_index(x, index=2, axis=1),
}
