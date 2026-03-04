from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.database import ECGRecord, Classification, get_db
from models.schemas import (
    ECGRecordCreate, 
    ECGRecordResponse,
    ECGRecordWithClassifications,
    ClassificationResponse
)


router = APIRouter(prefix="/records", tags=["ECG Records"])


@router.post("", response_model=ECGRecordResponse)
def create_record(record: ECGRecordCreate, db: Session = Depends(get_db)):
    """Crear un nuevo registro ECG"""
    db_record = ECGRecord(
        patient_id=record.patient_id,
        record_name=record.record_name,
        duration_seconds=record.duration_seconds,
        sampling_rate=record.sampling_rate,
        device_source=record.device_source,
        ecg_data=record.ecg_data,
        notes=record.notes
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record


@router.get("", response_model=List[ECGRecordResponse])
def get_records(
    patient_id: Optional[int] = None,
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Obtener lista de registros ECG"""
    query = select(ECGRecord)
    
    if patient_id:
        query = query.where(ECGRecord.patient_id == patient_id)
    
    records = db.execute(
        query.order_by(ECGRecord.created_at.desc()).offset(skip).limit(limit)
    ).scalars().all()
    return records


@router.get("/{record_id}", response_model=ECGRecordResponse)
def get_record(record_id: int, db: Session = Depends(get_db)):
    """Obtener un registro específico"""
    record = db.get(ECGRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    return record


@router.get("/{record_id}/classifications", response_model=List[ClassificationResponse])
def get_record_classifications(record_id: int, db: Session = Depends(get_db)):
    """Obtener clasificaciones de un registro"""
    classifications = db.execute(
        select(Classification)
        .where(Classification.record_id == record_id)
        .order_by(Classification.created_at.desc())
    ).scalars().all()
    return classifications


@router.delete("/{record_id}")
def delete_record(record_id: int, db: Session = Depends(get_db)):
    """Eliminar un registro ECG"""
    record = db.get(ECGRecord, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    db.delete(record)
    db.commit()
    return {"message": "Registro eliminado"}
