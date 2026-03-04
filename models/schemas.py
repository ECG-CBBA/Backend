from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from enum import Enum


class SexEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


# Patient Schemas
class PatientBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    sex: Optional[SexEnum] = None


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    sex: Optional[SexEnum] = None


class PatientResponse(PatientBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ECG Record Schemas
class ECGRecordBase(BaseModel):
    patient_id: Optional[int] = None
    record_name: Optional[str] = None
    duration_seconds: Optional[float] = None
    sampling_rate: Optional[int] = 360
    device_source: Optional[str] = "esp32"
    notes: Optional[str] = None


class ECGRecordCreate(ECGRecordBase):
    ecg_data: Optional[str] = None  # JSON string


class ECGRecordResponse(ECGRecordBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ECGRecordWithClassifications(ECGRecordResponse):
    classifications: List["ClassificationResponse"] = []


# Classification Schemas
class ClassificationBase(BaseModel):
    class_code: str
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: Optional[int] = None
    model_version: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())


class ClassificationCreate(ClassificationBase):
    record_id: Optional[int] = None
    probabilities: Optional[str] = None  # JSON string


class ClassificationResponse(ClassificationBase):
    id: int
    record_id: Optional[int]
    probabilities: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# WebSocket Classification Request/Response
class ECGDataRequest(BaseModel):
    type: str = "classify"
    session_id: str
    ecg_data: List[float]
    sampling_rate: int = 360
    metadata: dict = {}


class ClassificationResultResponse(BaseModel):
    type: str = "classification_result"
    session_id: str
    classification: str
    confidence: float
    arrhythmia_name: str
    processing_time_ms: int
    all_probabilities: dict = {}
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    timestamp: str

    model_config = ConfigDict(protected_namespaces=())


# Device Schemas
class DeviceBase(BaseModel):
    device_type: str
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    firmware_version: Optional[str] = None


class DeviceCreate(DeviceBase):
    pass


class DeviceResponse(DeviceBase):
    id: int
    is_active: bool
    last_seen: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
