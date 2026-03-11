from datetime import datetime
from dataclasses import dataclass
from typing import Dict

from preprocessing.ecg_processor import preprocess_ecg_data
from services.lstm_classifier import (
    LSTMClassifier,
)


@dataclass
class ClassificationResult:
    classification: str
    confidence: float
    class_name: str
    processing_time_ms: int
    all_probabilities: Dict[str, float]
    timestamp: str


def classify_ecg_segment(
    ecg_data: list,
    sampling_rate: int,
    classifier: LSTMClassifier,
) -> ClassificationResult:
    """
    Preprocesa y clasifica un segmento ECG.
    """
    if not ecg_data:
        raise ValueError("ecg_data no puede estar vacío.")

    processed = preprocess_ecg_data(ecg_data, sampling_rate)

    class_code, confidence, class_name, proc_time, all_probs = classifier.classify(
        processed
    )

    return ClassificationResult(
        classification=class_code,
        confidence=confidence,
        class_name=class_name,
        processing_time_ms=proc_time,
        all_probabilities=all_probs,
        timestamp=datetime.now().isoformat(),
    )
