import enum
import os
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    Enum as SQLEnum,
    create_engine,
)
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ecg.db")

engine = create_engine(
    DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    ),
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

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

    records = relationship(
        "ECGRecord", back_populates="patient", cascade="all, delete-orphan"
    )


class ECGRecord(Base):
    __tablename__ = "ecg_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    record_name = Column(String(100), index=True, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    sampling_rate = Column(Integer, default=360)
    device_source = Column(String(50), default="esp32")
    ecg_data = Column(Text, nullable=True)  # JSON string
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient = relationship("Patient", back_populates="records")
    classifications = relationship(
        "Classification", back_populates="record", cascade="all, delete-orphan"
    )


class Classification(Base):
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(Integer, ForeignKey("ecg_records.id"), nullable=True)
    class_code = Column(
        String(10), nullable=False
    )  # Normal, SVEB, VEB, Fusion, Unknown
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(Text, nullable=True)  # JSON string
    processing_time_ms = Column(Integer, nullable=True)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    record = relationship("ECGRecord", back_populates="classifications")


class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_type = Column(String(50), nullable=False)
    ip_address = Column(String(45), nullable=True)
    mac_address = Column(String(17), nullable=True)
    firmware_version = Column(String(20), nullable=True)
    is_active = Column(Integer, default=1)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    """
    provee una sesión de base de datos por request.
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(bind=engine)
