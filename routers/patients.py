from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from models.database import Patient, get_db
from models.schemas import (
    PatientCreate, 
    PatientUpdate, 
    PatientResponse
)


router = APIRouter(prefix="/patients", tags=["Patients"])


@router.post("", response_model=PatientResponse)
def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Crear un nuevo paciente"""
    db_patient = Patient(
        name=patient.name,
        email=patient.email,
        phone=patient.phone,
        date_of_birth=patient.date_of_birth,
        sex=patient.sex.value if patient.sex else None
    )
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("", response_model=List[PatientResponse])
def get_patients(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db)
):
    """Obtener lista de pacientes"""
    patients = db.execute(
        select(Patient).offset(skip).limit(limit)
    ).scalars().all()
    return patients


@router.get("/{patient_id}", response_model=PatientResponse)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Obtener un paciente específico"""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
def update_patient(
    patient_id: int, 
    patient: PatientUpdate, 
    db: Session = Depends(get_db)
):
    """Actualizar un paciente"""
    db_patient = db.get(Patient, patient_id)
    if not db_patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    
    for key, value in patient.model_dump(exclude_unset=True).items():
        setattr(db_patient, key, value)
    
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.delete("/{patient_id}")
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    """Eliminar un paciente"""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    db.delete(patient)
    db.commit()
    return {"message": "Paciente eliminado"}
