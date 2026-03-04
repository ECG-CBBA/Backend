from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()


class SexEnum(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    sex = Column(SQLEnum(SexEnum), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    records = relationship("ECGRecord", back_populates="patient", cascade="all, delete-orphan")


class ECGRecord(Base):
    __tablename__ = "ecg_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    # Record metadata
    record_name = Column(String(100), index=True, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sampling_rate = Column(Integer, default=360)
    device_source = Column(String(50), default="esp32")  # esp32, simulator
    
    # Data
    ecg_data = Column(Text, nullable=True)  # JSON string of ECG values
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="records")
    classifications = relationship("Classification", back_populates="record", cascade="all, delete-orphan")


class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(Integer, ForeignKey("ecg_records.id"), nullable=True)
    
    # Classification results
    class_code = Column(String(10), nullable=False)  # N, SVEB, VEB, Fusion, Unknown
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    
    # Probabilities for each class
    probabilities = Column(Text, nullable=True)  # JSON string
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    record = relationship("ECGRecord", back_populates="classifications")


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_type = Column(String(50), nullable=False)  # esp32, simulator
    ip_address = Column(String(45), nullable=True)
    mac_address = Column(String(17), nullable=True)
    firmware_version = Column(String(20), nullable=True)
    
    # Status
    is_active = Column(Integer, default=1)
    last_seen = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

def get_db():
    return Base