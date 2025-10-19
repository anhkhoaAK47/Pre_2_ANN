# app/state_store.py
# Lưu state in-memory theo user_id; sau này có thể thay bằng Redis/DB.

from typing import Dict, Any

_STATE: Dict[str, Dict[str, Any]] = {}  # user_id -> {slots, off_topic_count}

def get_state(user_id: str) -> Dict[str, Any]:
    return _STATE.setdefault(user_id, {"slots": {}, "off_topic_count": 0})

def merge_slots(user_id: str, new_slots: Dict[str, str]) -> Dict[str, str]:
    st = get_state(user_id)
    slots = st["slots"]
    for k, v in (new_slots or {}).items():
        if v:  # không ghi đè giá trị tốt bằng chuỗi rỗng
            slots[k] = v
    return slots

def update_fields(user_id: str, field_updates: Dict[str, str]) -> Dict[str, str]:
    """Update specific fields with new values (for corrections/modifications)"""
    st = get_state(user_id)
    slots = st["slots"]
    for field_name, new_value in (field_updates or {}).items():
        if field_name in ["name", "phone_number", "symptoms", "onset", "age", "gender", "allergies", "current_medications", "pain_scale"]:
            slots[field_name] = new_value
    return slots

def get_slots(user_id: str) -> Dict[str, str]:
    """Get only the slots part of the state"""
    return get_state(user_id)["slots"]

def increment_off_topic_count(user_id: str) -> int:
    """Increment and return the off-topic counter"""
    st = get_state(user_id)
    st["off_topic_count"] += 1
    return st["off_topic_count"]

def get_off_topic_count(user_id: str) -> int:
    """Get current off-topic count"""
    return get_state(user_id)["off_topic_count"]

def reset_state(user_id: str) -> None:
    _STATE.pop(user_id, None)
