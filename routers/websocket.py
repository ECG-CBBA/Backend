import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models.schemas import ECGDataRequest, ClassificationResultResponse
from services.lstm_classifier import (
    get_classifier,
    ModelNotLoadedError,
    ClassificationError,
)
from services.ecg_classification_service import classify_ecg_segment
from websocket.manager import manager


router = APIRouter(tags=["WebSocket"])


def _error_response(message: str, code: str) -> dict:
    return {
        "type": "error",
        "code": code,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para clasificación ECG en tiempo real."""
    await manager.connect(websocket)
    print("Cliente WebSocket conectado")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "classify":
                await _handle_classify(websocket, message)

            elif msg_type == "ping":
                await manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.now().isoformat()},
                    websocket,
                )

            else:
                await manager.send_personal_message(
                    _error_response(
                        f"Tipo de mensaje desconocido: '{msg_type}'",
                        "unknown_message_type",
                    ),
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)

    except Exception as e:
        print(f"Error en WebSocket: {e}")
        manager.disconnect(websocket)


async def _handle_classify(websocket: WebSocket, message: dict) -> None:
    """Orquesta una solicitud de clasificación y envía la respuesta."""
    try:
        request = ECGDataRequest(**message)

        result = classify_ecg_segment(
            ecg_data=request.ecg_data,
            sampling_rate=request.sampling_rate,
            classifier=get_classifier(),
        )

        response = ClassificationResultResponse(
            type="classification_result",
            session_id=request.session_id,
            classification=result.classification,
            confidence=result.confidence,
            arrhythmia_name=result.class_name,
            processing_time_ms=result.processing_time_ms,
            all_probabilities=result.all_probabilities,
            timestamp=result.timestamp,
        )

        await manager.send_personal_message(response.model_dump(), websocket)
        print(
            f"[classify] {result.classification} "
            f"({result.confidence:.2f}) — {result.processing_time_ms}ms"
        )

    except ModelNotLoadedError as e:
        # El modelo aún no fue cargado
        await manager.send_personal_message(
            _error_response(str(e), "model_not_loaded"),
            websocket,
        )

    except ClassificationError as e:
        # Falló la inferencia
        await manager.send_personal_message(
            _error_response(str(e), "classification_error"),
            websocket,
        )

    except ValueError as e:
        # Datos de entrada inválidos — error del cliente
        await manager.send_personal_message(
            _error_response(str(e), "invalid_input"),
            websocket,
        )
