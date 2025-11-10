from fastapi import FastAPI
from app.routers import items

app = FastAPI(title="Mi API FastAPI")

app.include_router(items.router)

@app.get("/")
def root():
    return {"mensaje": "Bienvenido a FastAPI 🚀"}
