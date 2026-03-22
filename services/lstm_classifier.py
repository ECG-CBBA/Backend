import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

CLASS_NAMES_AAMI = ["Normal", "SVEB", "VEB", "Fusion", "Unknown"]


class BiLSTMModel(nn.Module):
    """Arquitectura BiLSTM para clasificación de ECG"""

    def __init__(
        self,
        input_size=1,
        hidden_size_1=128,
        hidden_size_2=64,
        num_classes=5,
        dropout=0.3,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size, hidden_size_1, batch_first=True, bidirectional=True
        )
        self.bn1 = nn.BatchNorm1d(hidden_size_1 * 2)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(
            hidden_size_1 * 2, hidden_size_2, batch_first=True, bidirectional=True
        )
        self.bn2 = nn.BatchNorm1d(hidden_size_2 * 2)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2 * 2, 64)
        self.drop3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(64, 32)
        self.drop4 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(32, num_classes)

    def _bn_seq(self, bn, x):
        return bn(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(self._bn_seq(self.bn1, out))
        out, _ = self.lstm2(out)
        out = self.drop2(self._bn_seq(self.bn2, out))
        out = out[:, -1, :]
        out = self.drop3(torch.relu(self.fc1(out)))
        out = self.drop4(torch.relu(self.fc2(out)))
        return self.fc_out(out)


class LSTMClassifier:
    """Servicio de clasificación LSTM"""

    CLASS_MAPPING: Dict[int, str] = {
        0: "Normal",
        1: "SVEB",
        2: "VEB",
        3: "Fusion",
        4: "Unknown",
    }

    CLASS_NAMES: Dict[str, str] = {
        "Normal": "Ritmo sinusal normal",
        "SVEB": "Latido ectópico supraventricular",
        "VEB": "Latido ectópico ventricular",
        "Fusion": "Latido de fusión",
        "Unknown": "No clasificable / Marcapasos",
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model: Optional[BiLSTMModel] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or "bilstm_model.pth"
        self.window_size = 180
        self.model_version = "1.0.0"
        self.is_loaded = False

    def load_model(self) -> bool:
        """Cargar el modelo LSTM entrenado"""
        try:
            if os.path.exists(self.model_path):
                self.model = BiLSTMModel(
                    input_size=1,
                    hidden_size_1=128,
                    hidden_size_2=64,
                    num_classes=5,
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
        self.model = BiLSTMModel(
            input_size=1,
            hidden_size_1=128,
            hidden_size_2=64,
            num_classes=5,
        ).to(self.device)
        self.model.eval()
        self.is_loaded = False

    def classify(
        self, data: np.ndarray
    ) -> Tuple[str, float, str, int, Dict[str, float]]:

        if self.model is None:
            raise ModelNotLoadedError(
                "No hay modelo cargado. Llama a load_model() antes de clasificar."
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
                class_code = self.CLASS_MAPPING[predicted_class]
                class_name = self.CLASS_NAMES[class_code]

                all_probs = {
                    self.CLASS_NAMES[self.CLASS_MAPPING[i]]: float(
                        probabilities[0][i].item()
                    )
                    for i in range(5)
                }

                return (
                    class_code,
                    float(confidence.item()),
                    class_name,
                    processing_time,
                    all_probs,
                )

        except ModelNotLoadedError:
            raise
        except Exception as e:
            raise ClassificationError(f"Error durante la inferencia: {e}") from e

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


class ModelNotLoadedError(Exception):
    """Se lanza cuando se intenta clasificar sin modelo cargado."""

    pass


class ClassificationError(Exception):
    """Se lanza cuando falla el proceso de inferencia."""

    pass
