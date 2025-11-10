from pydantic import BaseModel

class Item(BaseModel):
    nombre: str
    descripcion: str | None = None
    precio: float
    disponible: bool = True
