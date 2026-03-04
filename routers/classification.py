import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.orm import Session

from models.schemas import (
    ECGDataRequest,
    ClassificationResultResponse,
    ClassificationCreate,
    ClassificationResponse,
)
from services.lstm_classifier import get_classifier
from preprocessing.ecg_processor import preprocess_ecg_data
from models.database import Classification, get_db


router = APIRouter(prefix="/classify", tags=["Classifications"])


@router.post("", response_model=ClassificationResultResponse)
async def classify_endpoint(request: ECGDataRequest):
    """Endpoint HTTP para clasificación de ECG"""
    try:
        processed_data = preprocess_ecg_data(request.ecg_data, request.sampling_rate)
        classifier = get_classifier()

        classification, confidence, class_name, processing_time, all_probs = (
            classifier.classify(processed_data)
        )

        return ClassificationResultResponse(
            type="classification_result",
            session_id=request.session_id,
            classification=classification,
            confidence=confidence,
            arrhythmia_name=class_name,
            processing_time_ms=processing_time,
            all_probabilities=all_probs,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/db", response_model=ClassificationResponse)
async def classify_and_save(
    classification: ClassificationCreate, db: Session = Depends(get_db)
):
    """Clasificar y guardar en base de datos"""
    db_classification = Classification(
        record_id=classification.record_id,
        class_code=classification.class_code,
        class_name=classification.class_name,
        confidence=classification.confidence,
        probabilities=classification.probabilities,
        processing_time_ms=classification.processing_time_ms,
        model_version=classification.model_version,
    )
    db.add(db_classification)
    db.commit()
    db.refresh(db_classification)
    return db_classification
