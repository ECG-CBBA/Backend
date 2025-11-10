from fastapi import APIRouter
from app.models.item_model import Item

router = APIRouter(prefix="/items", tags=["Items"])

items_db = []

@router.get("/")
def listar_items():
    return {"items": items_db}

@router.post("/")
def crear_item(item: Item):
    items_db.append(item)
    return {"mensaje": "Item agregado", "item": item}
