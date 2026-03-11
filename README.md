<!-- @format -->

# ECG Monitor Backend

FastAPI backend para el sistema de monitoreo de ECG en tiempo real con clasificación BiLSTM.

## Arquitectura

```
[AD8232] → [ESP32] → [WebSocket] → [FastAPI] → [BiLSTM Model]
                                              ↓
                                     [SQLite Database]
```

### Estructura del Proyecto

```
backend/
├── main.py
├── models/
│   ├── database.py                    # Configuración SQLAlchemy
│   └── schemas.py                     # Modelos Pydantic
├── routers/
│   ├── health.py                      # Endpoints de salud
│   ├── classification.py              # Clasificación HTTP
│   ├── websocket.py                   # Endpoint WebSocket en tiempo real
│   ├── patients.py                    # Gestión de pacientes
│   └── records.py                     # Registros ECG
├── services/
│   ├── lstm_classifier.py             # Clasificador BiLSTM + excepciones
│   └── ecg_classification_service.py  # Orquesta preprocesamiento + clasificación
├── preprocessing/
│   └── ecg_processor.py               # Preprocesamiento de señales ECG
└── websocket/
    └── manager.py                     # Gestor de conexiones WebSocket
```

---

## Protocolo de Comunicación (ESP32 → Backend)

### Paquete Binario (8 bytes)

| Offset | Tamaño | Campo     | Descripción              |
| ------ | ------ | --------- | ------------------------ |
| 0      | 4      | Timestamp | Tiempo en milisegundos   |
| 4      | 2      | ADC Value | Valor del ADC (0–4095)   |
| 6      | 1      | Quality   | Calidad de señal (0–100) |
| 7      | 1      | Battery   | Nivel de batería (0–100) |

### Mensajes WebSocket (JSON)

**Solicitud de clasificación:**

```json
{
    "type": "classify",
    "session_id": "session-123",
    "ecg_data": [0.1, 0.2, 0.15, "..."],
    "sampling_rate": 360,
    "metadata": {
        "device": "esp32",
        "patient_id": "opcional"
    }
}
```

**Respuesta de clasificación:**

```json
{
    "type": "classification_result",
    "session_id": "session-123",
    "classification": "VEB",
    "confidence": 0.91,
    "arrhythmia_name": "Latido ectópico ventricular",
    "processing_time_ms": 12,
    "all_probabilities": {
        "Ritmo sinusal normal": 0.04,
        "Latido ectópico supraventricular": 0.02,
        "Latido ectópico ventricular": 0.91,
        "Latido de fusión": 0.02,
        "No clasificable / Marcapasos": 0.01
    },
    "timestamp": "2026-01-15T10:30:00"
}
```

**Respuesta de error:**

```json
{
    "type": "error",
    "code": "model_not_loaded",
    "message": "No hay modelo cargado. Llama a load_model() antes de clasificar.",
    "timestamp": "2026-01-15T10:30:00"
}
```

**Códigos de error:**

| Código                 | Causa                                           |
| ---------------------- | ----------------------------------------------- |
| `model_not_loaded`     | El modelo no fue cargado al iniciar el servidor |
| `classification_error` | Error durante la inferencia del modelo          |
| `invalid_input`        | Datos ECG vacíos o con formato incorrecto       |
| `unknown_message_type` | El campo `type` del mensaje no es reconocido    |

---

## API Endpoints

### Health

| Método | Endpoint  | Descripción                   |
| ------ | --------- | ----------------------------- |
| GET    | `/`       | Información general de la API |
| GET    | `/health` | Estado del backend y modelo   |

### WebSocket

| Endpoint | Descripción                      |
| -------- | -------------------------------- |
| `/ws/ws` | Clasificación ECG en tiempo real |

### REST

| Método | Endpoint    | Descripción          |
| ------ | ----------- | -------------------- |
| GET    | `/patients` | Listar pacientes     |
| POST   | `/patients` | Crear paciente       |
| GET    | `/records`  | Listar registros ECG |
| POST   | `/records`  | Crear registro ECG   |
| POST   | `/classify` | Clasificación HTTP   |

---

## Clasificación — ANSI/AAMI EC57 (5 clases)

| Código    | Nombre clínico                      | Símbolos MIT-BIH |
| --------- | ----------------------------------- | ---------------- |
| `Normal`  | Ritmo sinusal normal                | N, L, R, e, j    |
| `SVEB`    | Latido ectópico supraventricular    | A, a, J, S       |
| `VEB`     | Latido ectópico ventricular         | V, E             |
| `Fusion`  | Latido de fusión ventricular-normal | F                |
| `Unknown` | No clasificable / Marcapasos        | /, f, Q, !       |

---

## Modelo BiLSTM

### Arquitectura

```
Input (batch, 180, 1)
    ↓
BiLSTM-1 [128 unidades, bidireccional] → BatchNorm → Dropout(0.3)
    ↓
BiLSTM-2 [64 unidades, bidireccional]  → BatchNorm → Dropout(0.3) → último timestep
    ↓
Dense 64 (ReLU) → Dropout(0.4)
    ↓
Dense 32 (ReLU) → Dropout(0.3)
    ↓
Dense 5 (logits)   ← CrossEntropyLoss incorpora Softmax internamente
```

### Parámetros

| Parámetro       | Valor                            |
| --------------- | -------------------------------- |
| Window size     | 180 muestras (0.5 s @ 360 Hz)    |
| Input size      | 1                                |
| Hidden BiLSTM-1 | 128 unidades (256 bidireccional) |
| Hidden BiLSTM-2 | 64 unidades (128 bidireccional)  |
| Dropout         | 0.3 (0.4 en Dense-1)             |
| Clases          | 5 (ANSI/AAMI EC57)               |

---

## Instalación

### Requisitos

- Python 3.12+
- pip
- Windows PowerShell

### Automática (recomendado)

```powershell
.\install.ps1
```

### Manual

```bash
python -m venv venv
venv\Scripts\activate                                          # Windows
pip install --upgrade pip setuptools wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Ejecución

```bash
# Desarrollo
fastapi dev main.py

# Producción
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Variables de Entorno

| Variable       | Default                      | Descripción             |
| -------------- | ---------------------------- | ----------------------- |
| `DATABASE_URL` | `sqlite:///./ecg_monitor.db` | URL de la base de datos |
| `MODEL_PATH`   | `bilstm_model.pth`           | Ruta del modelo BiLSTM  |

---

## Preprocesamiento de Señales

El ESP32 envía valores ADC (0–4095) que se convierten y normalizan antes de la inferencia:

```python
# 1. Conversión ADC → milivoltios
voltage_mv = (adc_value / 4095) * 3300  # referencia 3.3 V

# 2. Normalización z-score con el scaler entrenado (scaler.pkl)
#    El scaler fue ajustado solo sobre el conjunto de entrenamiento
sample_normalized = scaler.transform(sample)
```

---

## Errores Comunes

| Error                          | Causa                                                       | Solución                                                                                     |
| ------------------------------ | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Missing key(s) in state_dict` | Arquitectura del backend no coincide con el modelo guardado | Verificar que `ECG_BiLSTM` en `lstm_classifier.py` coincida con el notebook de entrenamiento |
| `model_not_loaded`             | Archivo `.pth` no encontrado                                | Verificar `MODEL_PATH`                                                                       |
| `WebSocket connection failed`  | Puerto bloqueado                                            | Verificar firewall                                                                           |
| `invalid_input`                | ECG enviado con menos de 180 muestras                       | Enviar exactamente 180 muestras                                                              |
