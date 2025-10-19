# app/models.py
# ORM models cho bảng patient_records

from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.sql import func
from app_v2.core.database import Base

class PatientRecord(Base):
    __tablename__ = "patient_records"

    record_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), index=True)
    
    # Patient Information
    patient_name = Column(String(255))
    patient_phone = Column(String(20))
    patient_age = Column(String(10))
    patient_gender = Column(String(10))
    
    # Medical Information
    symptoms = Column(Text)                 # JSON string of symptoms
    onset = Column(String(100))            # When symptoms started
    allergies = Column(Text)               # Allergies information
    current_medications = Column(Text)     # Current medications
    pain_scale = Column(String(10))        # Pain scale 1-10
    
    # Results
    predicted_diseases = Column(Text)       # JSON string of {label: prob}
    recommended_department = Column(String(128))
    
    # System
    chat = Column(Text)                     # optional: JSON string toàn bộ hội thoại/state
    created_at = Column(DateTime, default=func.now())
