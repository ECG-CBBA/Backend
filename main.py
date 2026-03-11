import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.database import Base, engine
from services.lstm_classifier import get_classifier
from routers import health, patients, records, classification, websocket

load_dotenv()


def init_db():
    Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Base de datos inicializada")

    classifier = get_classifier()
    print(f"Modelo {'cargado' if classifier.is_model_loaded else 'no cargado'}")

    yield
    print("Cerrando aplicación...")


app = FastAPI(
    title="ECG Monitor API",
    description="Backend para monitoreo ECG con clasificación LSTM",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(patients.router)
app.include_router(records.router)
app.include_router(classification.router)
app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        log_level="info"
    )