import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class ECG_BiLSTM(nn.Module):
    """Arquitectura BiLSTM para clasificación de ECG"""

    def __init__(
        self,
        input_size=1,
        hidden_size1=128,
        hidden_size2=64,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.lstm2 = nn.LSTM(
            input_size=hidden_size1 * 2,
            hidden_size=hidden_size2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size2 * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)

        out, _ = self.lstm2(out)
        out = self.dropout(out)

        out = out[:, -1, :]
        out = self.fc(out)

        return out


class LSTMClassifier:
    """Servicio de clasificación LSTM"""

    CLASS_MAPPING: Dict[int, str] = {
        0: "Normal",
        1: "Anormal",
    }

    CLASS_NAMES: Dict[str, str] = {
        "Normal": "Normal Sinus Rhythm",
        "Anormal": "Arrhythmia Detected",
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[ECG_BiLSTM] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "bilstm_model.pth"
        self.window_size = 180
        self.model_version = "1.0.0"
        self.is_loaded = False

    def load_model(self) -> bool:
        """Cargar el modelo LSTM entrenado"""
        try:
            if os.path.exists(self.model_path):
                self.model = ECG_BiLSTM(
                    input_size=1,
                    hidden_size1=128,
                    hidden_size2=64,
                    num_classes=2,
                ).to(self.device)

                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device)
                )
                self.model.eval()
                self.is_loaded = True
                print(f"Modelo BiLSTM cargado desde {self.model_path}")
                return True
            else:
                print(f"Modelo no encontrado en {self.model_path}")
                self._create_untrained_model()
                return False

        except Exception as e:
            print(f"Error cargando modelo: {e}")
            self._create_untrained_model()
            return False

    def _create_untrained_model(self):
        """Crear modelo sin entrenar para desarrollo"""
        self.model = ECG_BiLSTM(
            input_size=1,
            hidden_size1=128,
            hidden_size2=64,
            num_classes=2,
        ).to(self.device)
        self.model.eval()
        self.is_loaded = False

    def classify(
        self, data: np.ndarray
    ) -> Tuple[str, float, str, int, Dict[str, float]]:
        """Clasificar datos ECG"""
        if self.model is None:
            return (
                "Normal",
                0.75,
                "Normal Sinus Rhythm",
                0,
                {"Normal": 0.75, "Anormal": 0.25},
            )

        try:
            with torch.no_grad():
                x = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(self.device)

                start_time = datetime.now()
                outputs = self.model(x)
                processing_time = int(
                    (datetime.now() - start_time).total_seconds() * 1000
                )

                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                predicted_class = int(predicted.item())
                confidence_value = confidence.item()
                class_code = self.CLASS_MAPPING[predicted_class]
                class_name = self.CLASS_NAMES[class_code]

                all_probs = {
                    self.CLASS_NAMES[self.CLASS_MAPPING[i]]: float(
                        probabilities[0][i].item()
                    )
                    for i in range(2)
                }

                return (
                    class_code,
                    confidence_value,
                    class_name,
                    processing_time,
                    all_probs,
                )

        except Exception as e:
            print(f"Error en clasificación: {e}")
            return (
                "Normal",
                0.75,
                "Normal Sinus Rhythm",
                100,
                {"Normal": 0.75, "Anormal": 0.25},
            )

    @property
    def is_model_loaded(self) -> bool:
        return self.is_loaded


classifier_instance: Optional[LSTMClassifier] = None


def get_classifier() -> LSTMClassifier:
    """Obtener instancia global del clasificador"""
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = LSTMClassifier()
        classifier_instance.load_model()
    return classifier_instance
