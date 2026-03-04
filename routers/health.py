from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from models.schemas import HealthResponse
from services.lstm_classifier import get_classifier


router = APIRouter(tags=["Health"])


@router.get("/", tags=["Root"])
async def root():
    return {
        "message": "ECG Monitor API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint para verificar estado del backend"""
    classifier = get_classifier()
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.is_model_loaded,
        model_version=classifier.model_version,
        timestamp=datetime.now().isoformat()
    )
