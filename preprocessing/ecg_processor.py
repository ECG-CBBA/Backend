import pickle
import numpy as np
from typing import List

WINDOW_SIZE = 180
_scaler = None

def get_scaler():
    global _scaler
    if _scaler is None:
        with open("scaler.pkl", "rb") as f:
            _scaler = pickle.load(f)
    return _scaler

def preprocess_ecg_data(ecg_data: List[float], sampling_rate: int = 360) -> np.ndarray:
    data = np.array(ecg_data, dtype=np.float32)

    if len(data) < WINDOW_SIZE:
        data = np.pad(data, (0, WINDOW_SIZE - len(data)), 'constant')
    elif len(data) > WINDOW_SIZE:
        data = data[:WINDOW_SIZE]

    data = get_scaler().transform(data.reshape(1, -1)).flatten()

    return data.astype(np.float32)