import os
from datetime import datetime
import enum

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, Enum as SQLEnum, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = os.getenv("DB_PORT",     "5432")
DB_NAME     = os.getenv("DB_NAME",     "ecg_monitor")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class SexEnum(enum.Enum):
    MALE   = "male"
    FEMALE = "female"
    OTHER  = "other"


class Patient(Base):
    __tablename__ = "patients"

    id            = Column(Integer, primary_key=True, index=True)
    name          = Column(String(255), nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=True)
    phone         = Column(String(20),  nullable=True)
    date_of_birth = Column(DateTime,    nullable=True)
    sex           = Column(SQLEnum(SexEnum, name="sexenum"), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    records = relationship("ECGRecord", back_populates="patient", cascade="all, delete-orphan")


class ECGRecord(Base):
    __tablename__ = "ecg_records"

    id               = Column(Integer, primary_key=True, index=True)
    patient_id       = Column(Integer, ForeignKey("patients.id"), nullable=True)
    record_name      = Column(String(100), index=True, nullable=True)
    duration_seconds = Column(Float,       nullable=True)
    sampling_rate    = Column(Integer,     default=360)
    device_source    = Column(String(50),  default="esp32")
    ecg_data         = Column(Text,        nullable=True)
    notes            = Column(Text,        nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    patient         = relationship("Patient",        back_populates="records")
    classifications = relationship("Classification", back_populates="record", cascade="all, delete-orphan")


class Classification(Base):
    __tablename__ = "classifications"

    id                 = Column(Integer, primary_key=True, index=True)
    record_id          = Column(Integer, ForeignKey("ecg_records.id"), nullable=True)
    class_code         = Column(String(10),  nullable=False)
    class_name         = Column(String(100), nullable=False)
    confidence         = Column(Float,       nullable=False)
    probabilities      = Column(Text,        nullable=True)
    processing_time_ms = Column(Integer,     nullable=True)
    model_version      = Column(String(50),  nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow, index=True)

    record = relationship("ECGRecord", back_populates="classifications")


class Device(Base):
    __tablename__ = "devices"

    id               = Column(Integer, primary_key=True, index=True)
    device_type      = Column(String(50), nullable=False)
    ip_address       = Column(String(45), nullable=True)
    mac_address      = Column(String(17), nullable=True)
    firmware_version = Column(String(20), nullable=True)
    is_active        = Column(Boolean,  default=True)
    last_seen        = Column(DateTime, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()