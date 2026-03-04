from typing import List

import numpy as np


WINDOW_SIZE = 180


def preprocess_ecg_data(ecg_data: List[float], sampling_rate: int = 360) -> np.ndarray:
    """Preprocesar datos ECG para el modelo LSTM"""
    
    data = np.array(ecg_data, dtype=np.float32)
    
    if len(data) < WINDOW_SIZE:
        data = np.pad(data, (0, WINDOW_SIZE - len(data)), 'constant')
    elif len(data) > WINDOW_SIZE:
        data = data[:WINDOW_SIZE]
    
    data = np.clip(data, -5.0, 5.0)
    data = (data + 5.0) / 10.0
    
    try:
        from scipy import signal
        nyquist = sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        data = signal.filtfilt(b, a, data)
    except ImportError:
        pass
    
    data = np.clip(data, 0.0, 1.0)
    
    return data
