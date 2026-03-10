# ECG Monitor Backend

FastAPI backend para el sistema de monitoreo de ECG en tiempo real con clasificación LSTM.

## Arquitectura

```
[AD8232] → [ESP32] → [WebSocket] → [FastAPI] → [BiLSTM Model]
                                              ↓
                                     [SQLite Database]
```

## Estructura del Proyecto

```
backend/
├── main.py                      # Entry point de FastAPI
├── models/
│   ├── database.py             # Configuración SQLAlchemy
│   └── schemas.py              # Modelos Pydantic
├── routers/
│   ├── health.py               # Endpoints de salud
│   ├── classification.py       # Clasificación HTTP
│   ├── websocket.py            # Endpoint WebSocket en tiempo real
│   ├── patients.py             # Gestión de pacientes
│   └── records.py              # Registros ECG
├── services/
│   └── lstm_classifier.py     # Clasificador BiLSTM
├── preprocessing/
│   └── ecg_processor.py       # Preprocesamiento de datos ECG
└── websocket/
    └── manager.py              # Gestor de conexiones WebSocket
```

## Protocolo de Comunicación (ESP32 → Backend)

### Formato de Paquete Binario (8 bytes)

| Offset | Tamaño | Campo          | Descripción                    |
|--------|--------|----------------|--------------------------------|
| 0      | 4      | Timestamp      | Tiempo en milisegundos        |
| 4      | 2      | ADC Value      | Valor del ADC (0-4095)        |
| 6      | 1      | Quality        | Calidad de señal (0-100)     |
| 7      | 1      | Battery        | Nivel de batería (0-100)     |

### Formato JSON (WebSocket)

**Solicitud de clasificación:**
```json
{
  "type": "classify",
  "session_id": "session-123",
  "ecg_data": [0.1, 0.2, 0.15, ...],
  "sampling_rate": 360,
  "metadata": {
    "device": "esp32",
    "patient_id": "optional"
  }
}
```

**Respuesta de clasificación:**
```json
{
  "type": "classification_result",
  "session_id": "session-123",
  "classification": "Normal",
  "confidence": 0.95,
  "arrhythmia_name": "Normal Sinus Rhythm",
  "processing_time_ms": 12,
  "all_probabilities": {
    "Normal Sinus Rhythm": 0.95,
    "Arrhythmia Detected": 0.05
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## API Endpoints

### Health

| Método | Endpoint   | Descripción                    |
|--------|------------|--------------------------------|
| GET    | `/`        | Información general de la API |
| GET    | `/health`  | Estado del backend y modelo   |

### WebSocket

| Endpoint            | Descripción                           |
|---------------------|---------------------------------------|
| `/ws/ws`           | Clasificación en tiempo real         |

### REST (futuro)

| Método | Endpoint        | Descripción                    |
|--------|-----------------|--------------------------------|
| GET    | `/patients`     | Listar pacientes              |
| POST   | `/patients`     | Crear paciente                |
| GET    | `/records`      | Listar registros ECG          |
| POST   | `/records`      | Crear registro ECG            |
| POST   | `/classify`     | Clasificación sin WebSocket   |

## Clasificación

### Sistema Binario

| Código | Nombre   | Descripción                                    |
|--------|----------|------------------------------------------------|
| 0      | Normal   | Ritmo cardíaco sano, bloqueos de rama (N,L,R)|
| 1      | Anormal  | Arritmia que requiere atención (A,V,F,!,etc)  |

### Mapeo de Clases

| Símbolos MIT-BIH | Clase     |
|------------------|------------|
| N, L, R          | Normal     |
| A, J, S, V, E, F, /, f, Q, !, u | Anormal |

## Modelos LSTM

### Arquitectura BiLSTM

```
Input (180 samples)
    ↓
BiLSTM (128 hidden) × 2 directions
    ↓
Dropout (0.3)
    ↓
BiLSTM (64 hidden) × 2 directions
    ↓
Dropout (0.3)
    ↓
Fully Connected (2 classes)
    ↓
Softmax
```

### Parámetros

- Window size: 180 muestras
- Input size: 1 (una señal)
- Hidden size LSTM1: 128
- Hidden size LSTM2: 64
- Dropout: 0.3
- Clases: 2 (Normal/Anormal)

# Instalación

## Requisitos

- Python **3.12+**
- **pip**
- **Windows PowerShell**

---

## Instalación Automática (Recomendado)

Ejecuta el script:

```powershell
.\install.ps1
```

Este script realiza automáticamente:
* Creación del entorno virtual
* Activación del entorno
* Actualización de pip, setuptools y wheel
* Instalación de PyTorch (CPU)
* Instalación de dependencias del proyecto

Al finalizar verás:

```
✅ Instalación completa. Ejecuta: python main.py
```

## Instalación Manual (Alternativa)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

python -m pip install --upgrade pip setuptools wheel

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

## Ejecución

### Ejecutar el backend

```bash
python main.py
```

### Ejecutar con Uvicorn

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Variables de Entorno

| Variable      | Default              | Descripción              |
|---------------|---------------------|-------------------------|
| DATABASE_URL  | sqlite:///./ecg_monitor.db | URL de la base de datos |
| MODEL_PATH    | bilstm_model.pth    | Ruta del modelo LSTM    |

## Formato de Datos ECG

### Rango de Valores

El preprocesamiento normaliza los datos:
1. **Clipping**: [-5.0, 5.0]
2. **Normalización**: (data + 5.0) / 10.0 → [0, 1]
3. **Filtrado**: Butterworth bandpass [0.5, 40] Hz
4. **Padding/Truncation**: 180 muestras

### Simulación de Datos AD8232

El ESP32 envía valores ADC (0-4095) que deben convertirse a mV:
```python
# Conversión ADC a voltaje
voltage_mv = (adc_value / 4095) * 3300  # 3.3V referencia
```

## Errores Comunes

| Error                    | Causa                    | Solución               |
|--------------------------|--------------------------|-----------------------|
| Model not found          | Modelo no encontrado    | Verificar ruta MODEL_PATH |
| WebSocket connection fail| Puerto bloqueado        | Verificar firewall    |
| Classification timeout   | Datos insuficientes     | Enviar ≥180 muestras  |
