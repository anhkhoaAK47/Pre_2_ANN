# app/schemas.py
# Pydantic models cho JSON schema của LLM và payload /chat

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

# Slots bắt buộc cho flow (giữ đúng thứ tự để hỏi dần)
REQUIRED_SLOTS: List[str] = [
    "name", "phone_number", "symptoms", "onset", "age", "gender",
    "allergies", "current_medications", "pain_scale"
]

class LLMOutput(BaseModel):
    assistant_message: str = Field(..., description="Câu hiển thị cho người dùng (lịch sự, ngắn gọn)")
    slots_extracted: Dict[str, str] = Field(default_factory=dict)
    missing_slots: List[str] = Field(default_factory=list)
    next_action: str = Field(..., pattern=r"^(ask_for_missing_slots|call_phobert|final_confirmation|session_complete|off_topic_response|information_recall|information_updated)$")
    is_off_topic: bool = Field(default=False, description="True if this is an off-topic response")
    field_updates: Optional[Dict[str, str]] = Field(default=None, description="Field updates when patient corrects information")

    @validator("missing_slots", always=True)
    def only_required(cls, v):
        # Chỉ giữ những slot hợp lệ (tránh LLM trả thừa)
        return [s for s in v if s in REQUIRED_SLOTS]

    @validator("slots_extracted", always=True)
    def handle_null_slots(cls, v):
        # Convert null values to empty strings
        if v is None:
            return {}
        return {k: (v[k] if v[k] is not None else "") for k in v}

class ChatIn(BaseModel):
    user_id: str
    message: str

class ChatOut(BaseModel):
    assistant_message: str
    filled_slots: Dict[str, str]
    missing_slots: List[str]
    next_action: str
    debug: Optional[Dict] = None
