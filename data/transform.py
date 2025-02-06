import numpy as np

def select_index(data: np.ndarray, index: int = 0, axis: int = 0) -> np.ndarray:
    return np.squeeze(np.take(data, indices=[index], axis=axis))


TRANSFORM_MAP = {
    'direction_x': lambda x: select_index(x, index=0, axis=1),
    'direction_y': lambda x: select_index(x, index=1, axis=1),
    'face_length': lambda x: select_index(x, index=2, axis=1),
}
