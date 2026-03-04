import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from models.schemas import ECGDataRequest, ClassificationResultResponse
from services.lstm_classifier import get_classifier
from preprocessing.ecg_processor import preprocess_ecg_data
from websocket.manager import manager


router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para clasificación en tiempo real"""
    await manager.connect(websocket)
    print("Cliente WebSocket conectado")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "classify":
                try:
                    request = ECGDataRequest(**message)
                    
                    processed_data = preprocess_ecg_data(
                        request.ecg_data, 
                        request.sampling_rate
                    )
                    
                    classifier = get_classifier()
                    classification, confidence, class_name, processing_time, all_probs = classifier.classify(processed_data)
                    
                    response = ClassificationResultResponse(
                        type="classification_result",
                        session_id=request.session_id,
                        classification=classification,
                        confidence=confidence,
                        arrhythmia_name=class_name,
                        processing_time_ms=processing_time,
                        all_probabilities=all_probs,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    await manager.send_personal_message(response.model_dump(), websocket)
                    print(f"Clasificación: {classification} ({confidence:.2f}) - {processing_time}ms")
                    
                except Exception as e:
                    print(f"Error procesando clasificación: {e}")
                    error_response = {
                        "type": "error",
                        "message": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    await manager.send_personal_message(error_response, websocket)
                    
            elif message.get("type") == "ping":
                pong_response = {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_personal_message(pong_response, websocket)
                
    except WebSocketDisconnect:
        print("Cliente WebSocket desconectado")
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error en WebSocket: {e}")
        manager.disconnect(websocket)
