import pytest
import json
import numpy as np
import torch
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.ecg_processor import preprocess_ecg_data, WINDOW_SIZE
from models.schemas import ECGDataRequest, ClassificationResultResponse
from services.lstm_classifier import LSTMClassifier, ECG_BiLSTM


class TestECGPreprocessing:
    """Tests para el preprocesamiento de datos ECG"""

    def test_preprocess_ecg_normalizes_data(self):
        """Verifica que los datos se normalizan al rango [0, 1]"""
        ecg_data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        result = preprocess_ecg_data(ecg_data, sampling_rate=360)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_ecg_clips_extreme_values(self):
        """Verifica que valores extremos se cortan"""
        ecg_data = [-100.0, 0.0, 100.0]
        result = preprocess_ecg_data(ecg_data)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_ecg_pads_short_data(self):
        """Verifica que datos cortos se rellenan"""
        ecg_data = [0.5, 0.5, 0.5]
        result = preprocess_ecg_data(ecg_data)
        
        assert len(result) == WINDOW_SIZE

    def test_preprocess_ecg_truncates_long_data(self):
        """Verifica que datos largos se truncan"""
        ecg_data = [0.5] * 500
        result = preprocess_ecg_data(ecg_data)
        
        assert len(result) == WINDOW_SIZE

    def test_preprocess_ecg_returns_numpy_array(self):
        """Verifica que el resultado es un numpy array"""
        ecg_data = [0.5] * 180
        result = preprocess_ecg_data(ecg_data)
        
        assert isinstance(result, np.ndarray)


class TestECGDataRequest:
    """Tests para el esquema de datos ECG desde ESP32"""

    def test_ecg_data_request_valid(self):
        """Verifica creación válida de ECGDataRequest"""
        request = ECGDataRequest(
            type="classify",
            session_id="test-session-001",
            ecg_data=[0.1, 0.2, 0.3] * 60,
            sampling_rate=360,
            metadata={"device": "esp32"}
        )
        
        assert request.type == "classify"
        assert request.session_id == "test-session-001"
        assert len(request.ecg_data) == 180
        assert request.sampling_rate == 360

    def test_ecg_data_request_default_sampling_rate(self):
        """Verifica sampling rate por defecto"""
        request = ECGDataRequest(
            type="classify",
            session_id="test-session",
            ecg_data=[0.5] * 180
        )
        
        assert request.sampling_rate == 360


class TestAD8232Simulation:
    """Tests para simular datos del AD8232 enviados por ESP32"""

    @staticmethod
    def simulate_ad8232_packet(timestamp_ms: int, adc_value: int, 
                               quality: int, battery: int) -> bytes:
        """Simula un paquete de 8 bytes del ESP32"""
        packet = bytearray(8)
        
        packet[0:4] = timestamp_ms.to_bytes(4, 'little')
        packet[4:6] = adc_value.to_bytes(2, 'little')
        packet[6] = quality
        packet[7] = battery
        
        return bytes(packet)

    @staticmethod
    def adc_to_millivolts(adc_value: int, reference_mv: int = 3300) -> float:
        """Convierte valor ADC a milivoltios"""
        return (adc_value / 4095) * reference_mv

    def test_ad8232_packet_format(self):
        """Verifica formato del paquete de 8 bytes"""
        packet = self.simulate_ad8232_packet(
            timestamp_ms=1000,
            adc_value=2048,
            quality=95,
            battery=80
        )
        
        assert len(packet) == 8
        
        timestamp = int.from_bytes(packet[0:4], 'little')
        assert timestamp == 1000
        
        adc = int.from_bytes(packet[4:6], 'little')
        assert adc == 2048
        
        assert packet[6] == 95
        assert packet[7] == 80

    def test_ad8232_adc_conversion(self):
        """Verifica conversión ADC a milivoltios"""
        mv_zero = self.adc_to_millivolts(0)
        mv_mid = self.adc_to_millivolts(2048)
        mv_max = self.adc_to_millivolts(4095)
        
        assert mv_zero == 0.0
        assert 1600 < mv_mid < 1700
        assert mv_max > 3200

    def test_ecg_waveform_from_adc(self):
        """Simula una forma de onda ECG desde valores ADC"""
        sampling_rate = 360
        duration_seconds = 1
        num_samples = sampling_rate * duration_seconds
        
        t = np.linspace(0, duration_seconds, num_samples)
        heart_rate = 60
        frequency = heart_rate / 60
        
        ecg_waveform = (
            np.sin(2 * np.pi * frequency * t) * 500 +
            np.sin(2 * np.pi * frequency * 3 * t) * 200 +
            np.sin(2 * np.pi * frequency * 5 * t) * 100
        )
        
        adc_values = (ecg_waveform + 2048).astype(int)
        adc_values = np.clip(adc_values, 0, 4095)
        
        assert len(adc_values) == 360
        assert adc_values.min() >= 0
        assert adc_values.max() <= 4095


class TestLSTMClassifier:
    """Tests para el clasificador LSTM"""

    @patch('services.lstm_classifier.ECG_BiLSTM')
    def test_classifier_initialization(self, mock_model):
        """Verifica inicialización del clasificador"""
        classifier = LSTMClassifier(model_path="test_model.pth")
        
        assert classifier.model_path == "test_model.pth"
        assert classifier.window_size == 180
        assert classifier.model is None

    def test_classifier_returns_default_when_no_model(self):
        """Verifica respuesta por defecto sin modelo"""
        classifier = LSTMClassifier()
        
        result = classifier.classify(np.random.rand(180))
        
        assert result[0] == "Normal"
        assert result[1] == 0.75
        assert "Normal" in result[2]

    def test_class_mapping(self):
        """Verifica mapeo de clases"""
        classifier = LSTMClassifier()
        
        assert classifier.CLASS_MAPPING[0] == "Normal"
        assert classifier.CLASS_MAPPING[1] == "Anormal"

    def test_class_names(self):
        """Verifica nombres de clases"""
        classifier = LSTMClassifier()
        
        assert classifier.CLASS_NAMES["Normal"] == "Normal Sinus Rhythm"
        assert classifier.CLASS_NAMES["Anormal"] == "Arrhythmia Detected"


class TestBiLSTMArchitecture:
    """Tests para la arquitectura BiLSTM"""

    def test_bilstm_forward_pass(self):
        """Verifica forward pass de la red"""
        model = ECG_BiLSTM(
            input_size=1,
            hidden_size1=128,
            hidden_size2=64,
            num_classes=2
        )
        
        batch_size = 1
        sequence_length = 180
        input_data = torch.randn(batch_size, sequence_length, 1)
        
        output = model(input_data)
        
        assert output.shape == (batch_size, 2)

    def test_bilstm_output_range(self):
        """Verifica que la salida tiene las dimensiones correctas"""
        model = ECG_BiLSTM()
        
        x = torch.randn(1, 180, 1)
        output = model(x)
        
        assert output.shape == (1, 2)


class TestWebSocketProtocol:
    """Tests para el protocolo WebSocket"""

    def test_classification_result_response(self):
        """Verifica respuesta de clasificación"""
        response = ClassificationResultResponse(
            type="classification_result",
            session_id="session-123",
            classification="Normal",
            confidence=0.95,
            arrhythmia_name="Normal Sinus Rhythm",
            processing_time_ms=15,
            all_probabilities={
                "Normal Sinus Rhythm": 0.95,
                "Arrhythmia Detected": 0.05
            },
            timestamp="2024-01-15T10:30:00"
        )
        
        assert response.type == "classification_result"
        assert response.classification == "Normal"
        assert response.confidence == 0.95

    def test_classification_result_to_dict(self):
        """Verifica conversión a diccionario"""
        response = ClassificationResultResponse(
            type="classification_result",
            session_id="test",
            classification="Normal",
            confidence=0.9,
            arrhythmia_name="Normal",
            processing_time_ms=10,
            all_probabilities={},
            timestamp="2024-01-01"
        )
        
        result_dict = response.model_dump()
        
        assert isinstance(result_dict, dict)
        assert "classification" in result_dict
        assert "confidence" in result_dict


class TestEndToEndFlow:
    """Tests de flujo completo ESP32 → Backend"""

    def test_full_ecg_pipeline(self):
        """Test de flujo completo: ADC → Normalización → Clasificación"""
        
        raw_adc_values = np.random.randint(1500, 2500, size=360)
        
        ecg_mv = (raw_adc_values / 4095) * 3300
        ecg_normalized = (ecg_mv - ecg_mv.min()) / (ecg_mv.max() - ecg_mv.min() + 1e-8)
        
        preprocessed = preprocess_ecg_data(ecg_normalized.tolist())
        
        assert preprocessed.shape == (180,)
        assert preprocessed.min() >= 0.0
        assert preprocessed.max() <= 1.0
        
        classifier = LSTMClassifier()
        classification, confidence, name, time, probs = classifier.classify(preprocessed)
        
        assert classification in ["Normal", "Anormal"]
        assert 0.0 <= confidence <= 1.0
        assert isinstance(time, int)

    def test_session_message_format(self):
        """Verifica formato de mensaje de sesión"""
        request = ECGDataRequest(
            type="classify",
            session_id="esp32-session-001",
            ecg_data=[0.5] * 180,
            sampling_rate=360,
            metadata={
                "device": "esp32",
                "firmware": "1.0.0",
                "battery": 85,
                "signal_quality": 90
            }
        )
        
        message_dict = request.model_dump()
        
        assert message_dict["type"] == "classify"
        assert "esp32-session" in message_dict["session_id"]
        assert len(message_dict["ecg_data"]) == 180
        assert message_dict["metadata"]["device"] == "esp32"


class TestConnectionManager:
    """Tests para el gestor de conexiones WebSocket"""

    def test_connection_manager_init(self):
        """Verifica inicialización del gestor"""
        from websocket.manager import ConnectionManager
        
        manager = ConnectionManager()
        
        assert manager.connection_count == 0
        assert len(manager.active_connections) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
