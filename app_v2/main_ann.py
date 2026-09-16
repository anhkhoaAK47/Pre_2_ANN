"""
Vietnamese Medical Chatbot V2 with ANN and Active Questioning
- Uses PyTorch ANN model for 577-disease classification
- Implements advanced active questioning with information gain
- Progressive symptom collection through intelligent dialogue
- Confidence-based decision making (HIGH/MEDIUM/LOW)
- Systematic department mapping and database storage

Author: AI Medical Chatbot Team
Date: 2025-10-18
Version: 2.0 with ANN Active Questioning
"""

# ========= IMPORTS =========
import re
import os
import json
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

import pandas as pd
import torch

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy.orm import Session
from app_v2.core.database import get_db
from app_v2.models.database_models import PatientRecord

# Import ANN services
from app_v2.services.ann_service import ANNService, PredictionResult
from app_v2.services.active_questioning import (
    QuestioningSession,
    Question,
    Hypothesis,
    QuestioningStrategy
)

# Import conversation components
from app_v2.schemas.chat_schemas import ChatIn, ChatOut, REQUIRED_SLOTS
from app_v2.core.state_store import (
    merge_slots, get_slots, get_off_topic_count,
    increment_off_topic_count, update_fields
)
from app_v2.services.gemini_service import call_gemini

# ========= CONFIGURATION =========
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8004"))  # New port for v2

# ANN Model Configuration
# BASE_DIR points to Pre_2_ANN directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "building_ANN", "models", "pytorch")
DATA_DIR = os.path.join(BASE_DIR, "building_ANN", "data", "processed", "common")
EVAL_DIR = os.path.join(BASE_DIR, "building_ANN", "data", "evaluation")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
CONFUSION_MATRIX_PATH = os.path.join(EVAL_DIR, "most_confused_pairs.csv")

# Department Mapping Configuration
DEPARTMENT_DEFAULT = os.getenv("DEPARTMENT_DEFAULT", "Khám tổng quát")
# Use the comprehensive 577-disease mapping from building_ANN/utils
DEPARTMENT_MAP_PATH = os.getenv(
    "DEPARTMENT_MAP_PATH",
    os.path.join(BASE_DIR, "building_ANN", "utils", "department_map_577.json")
)

# ========= DEPARTMENT MAPPING =========
def load_department_map(path: str) -> dict:
    try:
        if not os.path.exists(path):
            logger.warning(f"Department map not found at {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading department map: {e}")
        return {}

DEPT_MAP = load_department_map(DEPARTMENT_MAP_PATH)

# Regex rules for disease-to-department mapping
REGEX_RULES = [
    (r"(?i)ung\s*thư", "Ung bướu"),
    (r"(?i)viêm\s*gan", "Tiêu hoá – Gan mật"),
    (r"(?i)gan\s*to", "Tiêu hoá – Gan mật"),
    (r"(?i)trào\s*ngược|GERD", "Tiêu hoá"),
    (r"(?i)dạ\s*dày|ruột|loét", "Tiêu hoá"),
    (r"(?i)tiêu\s*chảy|ngộ\s*độc", "Tiêu hoá"),
    (r"(?i)hen|viêm\s*phổi|khó\s*thở|cảm\s*lạnh", "Nội hô hấp"),
    (r"(?i)covid", "Sàng lọc COVID / Truyền nhiễm"),
    (r"(?i)sốt\s*xuất\s*huyết|sốt\s*rét|thương\s*hàn|AIDS|thủy\s*đậu|truyền\s*nhiễm", "Truyền nhiễm"),
    (r"(?i)đau\s*tim|tăng\s*huyết\s*áp|cao\s*huyết\s*áp|mạch|bẩm\s*sinh\s*tim", "Tim mạch"),
    (r"(?i)tiểu\s*đường|đái\s*tháo\s*đường|cường\s*giáp|suy\s*giáp|hạ\s*đường\s*huyết", "Nội tiết"),
    (r"(?i)thiếu\s*máu", "Huyết học"),
    (r"(?i)viêm\s*khớp|xương\s*khớp", "Cơ xương khớp"),
    (r"(?i)gãy|rách|chấn\s*thương|đầu\s*gối|mắt\s*cá", "Chấn thương chỉnh hình"),
    (r"(?i)nhiễm\s*trùng\s*đường\s*tiết\s*niệu|thận", "Thận – Tiết niệu"),
    (r"(?i)âm\s*đạo|thai\s*kỳ|sản", "Sản phụ khoa"),
    (r"(?i)mụn|vẩy\s*nến|gàu|chốc\s*lở|da\s*liễu", "Da liễu"),
    (r"(?i)đau\s*nửa\s*đầu|chóng\s*mặt|mất\s*thăng\s*bằng|thần\s*kinh", "Thần kinh"),
    (r"(?i)tràn\s*khí\s*màng\s*phổi|tắc\s*ruột|xuất\s*huyết", "Cấp cứu")
]

def recommend_department(disease_name: str) -> str:
    """Map predicted disease to appropriate medical department"""
    if not disease_name:
        return DEPARTMENT_DEFAULT

    # Try exact match first
    if disease_name in DEPT_MAP:
        return DEPT_MAP[disease_name]

    # Try case-insensitive match
    disease_lower = disease_name.lower().strip()
    for k, v in DEPT_MAP.items():
        if k.lower().strip() == disease_lower:
            return v

    # Try regex pattern matching
    for pattern, dept in REGEX_RULES:
        if re.search(pattern, disease_lower):
            return dept

    # Default fallback
    return DEPARTMENT_DEFAULT

# ========= DEVICE CONFIGURATION =========
def pick_device():
    """Select the best available device for PyTorch"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = pick_device()
logger.info(f"Using device: {DEVICE}")

# ========= INITIALIZE ANN SERVICE =========
try:
    logger.info("Initializing ANN Service...")
    ann_service = ANNService(
        model_path=MODEL_PATH,
        label_encoder_path=LABEL_ENCODER_PATH,
        train_data_path=TRAIN_DATA_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH if os.path.exists(CONFUSION_MATRIX_PATH) else None,
        department_map=DEPT_MAP,  # Enable department-based questioning
        device=DEVICE
    )
    logger.info("✅ ANN Service initialized successfully")
    logger.info("✅ Department-based questioning enabled")
except Exception as e:
    logger.error(f"❌ Failed to initialize ANN Service: {e}")
    raise

# ========= FASTAPI APPLICATION =========
app = FastAPI(
    title="Medical Chatbot V2 - ANN with Active Questioning",
    version="2.0",
    description="Advanced medical chatbot using PyTorch ANN and information-gain based active questioning"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= API MODELS =========
class PredictSymptomsIn(BaseModel):
    symptoms: List[str]
    top_k: int = 10

class DiseasePredict(BaseModel):
    disease: str
    probability: float

class PredictSymptomsOut(BaseModel):
    top_disease: str
    confidence: float
    confidence_category: str
    needs_questioning: bool
    top_predictions: List[DiseasePredict]
    department: str

class QuestionResponse(BaseModel):
    question_text: str
    symptom_name: str
    reason: str
    questions_asked: int
    max_questions: int

class AnswerQuestion(BaseModel):
    session_id: str
    has_symptom: bool

# ========= API ENDPOINTS =========
@app.get("/health")
def health():
    """System health check"""
    return {
        "status": "ok",
        "version": "2.0",
        "device": str(DEVICE),
        "model": "PyTorch ANN",
        "num_diseases": ann_service.num_classes,
        "num_symptoms": ann_service.num_symptoms,
        "active_sessions": len(ann_service.active_sessions)
    }

@app.post("/predict_symptoms", response_model=PredictSymptomsOut)
def predict_symptoms(inp: PredictSymptomsIn):
    """
    Direct disease prediction from symptom list (no active questioning).
    Use this for quick predictions without interactive dialogue.
    """
    try:
        prediction = ann_service.predict_from_symptoms(inp.symptoms, top_k=inp.top_k)
        dept = recommend_department(prediction.top_disease)

        return PredictSymptomsOut(
            top_disease=prediction.top_disease,
            confidence=prediction.top_confidence,
            confidence_category=prediction.confidence_category,
            needs_questioning=prediction.needs_questioning,
            top_predictions=[
                {"disease": d, "probability": float(p)}
                for d, p in prediction.top_k_predictions
            ],
            department=dept
        )
    except Exception as e:
        logger.error(f"Error in predict_symptoms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn, db: Session = Depends(get_db)):
    """
    Main conversational endpoint with ANN and active questioning:
    - Collects patient information through Gemini dialogue
    - Uses ANN model for disease prediction
    - Implements active questioning for ambiguous cases (disease or department mode)
    - Provides department recommendations
    - Saves consultations to database

    Query Parameters:
    - mode: 'disease' (default) or 'department' - controls questioning approach
    """
    user_id = body.user_id
    message = body.message
    mode = body.mode or "disease"  # Default to disease-based

    try:
        # 1) Get current slots
        slots = get_slots(user_id)
        off_topic_count = get_off_topic_count(user_id)

        # 2) Call Gemini LLM for slot filling
        from app_v2.schemas.chat_schemas import LLMOutput
        llm_out = call_gemini(message, slots, off_topic_count)
        parsed = LLMOutput(**llm_out)

        # 3) Handle field updates
        if parsed.field_updates:
            filled = update_fields(user_id, parsed.field_updates)
        else:
            filled = merge_slots(user_id, parsed.slots_extracted)

        # 4) Calculate missing slots
        missing = [s for s in REQUIRED_SLOTS if not filled.get(s)]

        # 5) Handle off-topic
        if parsed.next_action == "off_topic_response" or parsed.is_off_topic:
            new_count = increment_off_topic_count(user_id)
            if new_count >= 2:
                assist_msg = parsed.assistant_message
                next_action = "guide_back_to_medical"
            else:
                assist_msg = parsed.assistant_message
                next_action = "off_topic_response"

        # 6) Handle information recall
        elif parsed.next_action == "information_recall":
            assist_msg = parsed.assistant_message
            next_action = "information_recall"

        # 7) Handle information updates
        elif parsed.next_action == "information_updated":
            assist_msg = parsed.assistant_message
            next_action = "information_updated"

            if not missing:
                _update_patient_record(db, user_id, filled, missing, None, None)

        # 8) Main decision: Call ANN when ready
        elif parsed.next_action == "call_phobert" and not missing:
            # Get symptoms
            symptoms_text = filled.get("symptoms", "")

            # Extract symptom list (simple approach - you can enhance this)
            # For now, we'll use the raw symptoms text and let the user provide symptom names
            # In a real system, you'd parse the text to extract symptom names

            # Parse symptoms from text (simplified - in production you'd use NER)
            symptom_list = _extract_symptoms_from_text(symptoms_text)

            if not symptom_list:
                assist_msg = (
                    "Tôi chưa nhận diện được triệu chứng rõ ràng. "
                    "Bạn có thể mô tả chi tiết hơn các triệu chứng của mình không?"
                )
                next_action = "ask_for_symptoms"
            else:
                # Make initial prediction
                prediction = ann_service.predict_from_symptoms(symptom_list, top_k=10)
                dept = recommend_department(prediction.top_disease)

                # Check if we need active questioning (MODE-DEPENDENT)
                if prediction.needs_questioning and prediction.confidence_category != "HIGH":
                    # Start questioning session based on mode
                    if mode == "department":
                        # DEPARTMENT-BASED questioning
                        session_id = f"dept_aq_{user_id}_{uuid.uuid4().hex[:8]}"
                        session, first_question = ann_service.start_department_questioning_session(
                            session_id=session_id,
                            initial_symptoms=symptom_list
                        )

                        if first_question:
                            # Get top departments for display
                            top_depts = [h.department_name for h in session.current_departments[:3]]

                            # Store session ID in user's state
                            assist_msg = (
                                f"🔍 **Đang phân tích triệu chứng...**\n\n"
                                f"📊 Độ tin cậy hiện tại: {session.initial_confidence*100:.1f}%\n"
                                f"🏥 **Khoa có khả năng:** {', '.join(top_depts)}\n\n"
                                f"❓ **Câu hỏi {session.questions_asked + 1}/{session.max_questions}:**\n"
                                f"{first_question.question_text}\n\n"
                                f"💡 *{first_question.reason}*\n\n"
                                f"Vui lòng trả lời: **Có** hoặc **Không**"
                            )
                            next_action = "active_questioning_department"

                            # Save session reference
                            filled["_active_session_id"] = session_id
                            filled["_current_question_symptom"] = first_question.symptom_name
                            filled["_questioning_mode"] = "department"
                        else:
                            # No good question found, proceed with current prediction
                            assist_msg = _format_diagnosis_result(prediction, dept)
                            next_action = "final_confirmation"
                            _update_patient_record(db, user_id, filled, missing, prediction, dept)

                    else:  # mode == "disease"
                        # DISEASE-BASED questioning
                        session_id = f"aq_{user_id}_{uuid.uuid4().hex[:8]}"
                        session, first_question = ann_service.start_questioning_session(
                            session_id=session_id,
                            initial_symptoms=symptom_list
                        )

                        if first_question:
                            # Get top diseases for display
                            top_diseases = [h.disease_name for h in session.current_hypotheses[:3]]

                            # Store session ID in user's state
                            assist_msg = (
                                f"🔍 **Đang phân tích triệu chứng...**\n\n"
                                f"📊 Độ tin cậy hiện tại: {session.initial_confidence*100:.1f}%\n"
                                f"🩺 **Bệnh có khả năng:** {', '.join(top_diseases)}\n\n"
                                f"❓ **Câu hỏi {session.questions_asked + 1}/{session.max_questions}:**\n"
                                f"{first_question.question_text}\n\n"
                                f"💡 *{first_question.reason}*\n\n"
                                f"Vui lòng trả lời: **Có** hoặc **Không**"
                            )
                            next_action = "active_questioning_disease"

                            # Save session reference
                            filled["_active_session_id"] = session_id
                            filled["_current_question_symptom"] = first_question.symptom_name
                            filled["_questioning_mode"] = "disease"
                        else:
                            # No good question found, proceed with current prediction
                            assist_msg = _format_diagnosis_result(prediction, dept)
                            next_action = "final_confirmation"
                            _update_patient_record(db, user_id, filled, missing, prediction, dept)
                else:
                    # High confidence, no questioning needed
                    assist_msg = _format_diagnosis_result(prediction, dept)
                    next_action = "final_confirmation"
                    _update_patient_record(db, user_id, filled, missing, prediction, dept)

        # 9) Handle active questioning responses (MODE-DEPENDENT)
        elif parsed.next_action == "active_questioning_response":
            session_id = filled.get("_active_session_id")
            current_symptom = filled.get("_current_question_symptom")
            questioning_mode = filled.get("_questioning_mode", "disease")  # Default to disease

            if not session_id or not current_symptom:
                assist_msg = "Xin lỗi, có lỗi trong quá trình đặt câu hỏi. Vui lòng thử lại."
                next_action = "error"
            else:
                # Parse user's yes/no answer
                has_symptom = _parse_yes_no(message)

                if has_symptom is None:
                    assist_msg = "Xin vui lòng trả lời **Có** hoặc **Không** cho câu hỏi trước."
                    next_action = f"active_questioning_{questioning_mode}"
                else:
                    if questioning_mode == "department":
                        # Process answer using DEPARTMENT-BASED questioning
                        session, next_question = ann_service.answer_department_question(
                            session_id=session_id,
                            symptom_name=current_symptom,
                            has_symptom=has_symptom
                        )

                        if session.is_complete or not next_question:
                            # Questioning complete - show department result
                            final_department = session.final_department

                            # Get top disease from final department's contributing diseases
                            top_disease_name = final_department.contributing_diseases[0][0]
                            top_disease_prob = final_department.contributing_diseases[0][1]

                            assist_msg = (
                                f"✅ **Phân tích hoàn tất!**\n\n"
                                f"🏥 **Kết quả khám:**\n"
                                f"• Khoa khám gợi ý: **{final_department.department_name}**\n"
                                f"• Độ tin cậy khoa: **{final_department.probability*100:.1f}%**\n"
                                f"• Bệnh có khả năng cao nhất: **{top_disease_name}** ({top_disease_prob*100:.1f}%)\n\n"
                                f"📊 Số câu hỏi đã hỏi: {session.questions_asked}/{session.max_questions}\n\n"
                                f"📋 Bạn có thể in hồ sơ tư vấn để mang đến bệnh viện.\n\n"
                                f"❓ Bạn có muốn hỏi thêm điều gì khác không? Nếu không, hãy nhập 'kết thúc' để hoàn tất."
                            )
                            next_action = "final_confirmation"

                            # Save to database - build symptom list from vector
                            all_symptoms = []
                            for symptom_idx, symptom_val in enumerate(session.current_symptom_vector):
                                if symptom_val == 1:
                                    all_symptoms.append(ann_service.symptom_columns[symptom_idx])

                            prediction = ann_service.predict_from_symptoms(all_symptoms, top_k=10)
                            _update_patient_record(db, user_id, filled, missing, prediction, final_department.department_name)

                            # Clean up session
                            filled.pop("_active_session_id", None)
                            filled.pop("_current_question_symptom", None)
                            filled.pop("_questioning_mode", None)

                        else:
                            # Continue questioning - show DEPARTMENT progress
                            current_top_dept = session.current_departments[0]
                            assist_msg = (
                                f"📊 **Cập nhật phân tích:**\n"
                                f"• Khoa có khả năng cao nhất: **{current_top_dept.department_name}** ({current_top_dept.probability*100:.1f}%)\n\n"
                                f"❓ **Câu hỏi {session.questions_asked + 1}/{session.max_questions}:**\n"
                                f"{next_question.question_text}\n\n"
                                f"💡 *{next_question.reason}*\n\n"
                                f"Vui lòng trả lời: **Có** hoặc **Không**"
                            )
                            next_action = "active_questioning_department"

                            # Update current question
                            filled["_current_question_symptom"] = next_question.symptom_name

                    else:  # questioning_mode == "disease"
                        # Process answer using DISEASE-BASED questioning
                        session, next_question = ann_service.answer_question(
                            session_id=session_id,
                            symptom_name=current_symptom,
                            has_symptom=has_symptom
                        )

                        if session.is_complete or not next_question:
                            # Questioning complete - show disease result
                            final_disease = session.final_hypothesis
                            dept = recommend_department(final_disease.disease_name)

                            assist_msg = (
                                f"✅ **Phân tích hoàn tất!**\n\n"
                                f"🩺 **Kết quả khám:**\n"
                                f"• Bệnh có khả năng: **{final_disease.disease_name}**\n"
                                f"• Độ tin cậy: **{final_disease.probability*100:.1f}%**\n"
                                f"• Khoa khám gợi ý: **{dept}**\n\n"
                                f"📊 Số câu hỏi đã hỏi: {session.questions_asked}/{session.max_questions}\n\n"
                                f"📋 Bạn có thể in hồ sơ tư vấn để mang đến bệnh viện.\n\n"
                                f"❓ Bạn có muốn hỏi thêm điều gì khác không? Nếu không, hãy nhập 'kết thúc' để hoàn tất."
                            )
                            next_action = "final_confirmation"

                            # Save to database - build symptom list from vector
                            all_symptoms = []
                            for symptom_idx, symptom_val in enumerate(session.current_symptom_vector):
                                if symptom_val == 1:
                                    all_symptoms.append(ann_service.symptom_columns[symptom_idx])

                            prediction = ann_service.predict_from_symptoms(all_symptoms, top_k=10)
                            _update_patient_record(db, user_id, filled, missing, prediction, dept)

                            # Clean up session
                            filled.pop("_active_session_id", None)
                            filled.pop("_current_question_symptom", None)
                            filled.pop("_questioning_mode", None)

                        else:
                            # Continue questioning - show DISEASE progress
                            current_top_disease = session.current_hypotheses[0]
                            assist_msg = (
                                f"📊 **Cập nhật phân tích:**\n"
                                f"• Bệnh có khả năng cao nhất: **{current_top_disease.disease_name}** ({current_top_disease.probability*100:.1f}%)\n\n"
                                f"❓ **Câu hỏi {session.questions_asked + 1}/{session.max_questions}:**\n"
                                f"{next_question.question_text}\n\n"
                                f"💡 *{next_question.reason}*\n\n"
                                f"Vui lòng trả lời: **Có** hoặc **Không**"
                            )
                            next_action = "active_questioning_disease"

                            # Update current question
                            filled["_current_question_symptom"] = next_question.symptom_name

        # 10) Session complete
        elif parsed.next_action == "session_complete" or message.lower().strip() in [
            'kết thúc', 'ket thuc', 'end', 'finish', 'hoàn tất', 'hoan tat'
        ]:
            assist_msg = (
                f"🙏 **Cảm ơn bạn đã sử dụng dịch vụ tư vấn y tế AI V2!**\n\n"
                f"📋 Hồ sơ tư vấn đã được lưu và sẵn sàng để in.\n"
                f"🏥 Vui lòng đến bệnh viện để được thăm khám chính xác.\n\n"
                f"🔄 **Để bắt đầu phiên mới, nhấn nút 'Phiên mới'.**\n\n"
                f"⚠️ *Lưu ý: Thông tin hiện tại sẽ được giữ để bạn có thể in hồ sơ.*"
            )
            next_action = "session_complete"

        # 11) Default: ask for missing slots
        else:
            assist_msg = parsed.assistant_message
            next_action = "ask_for_missing_slots"

        return ChatOut(
            assistant_message=assist_msg,
            filled_slots=filled,
            missing_slots=missing,
            next_action=next_action,
            mode=mode,
            debug={
                "llm_next_action": parsed.next_action,
                "off_topic_count": get_off_topic_count(user_id),
                "is_off_topic": parsed.is_off_topic,
                "model_version": "ANN_v2",
                "questioning_mode": mode
            }
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatOut(
            assistant_message="Xin lỗi, có lỗi xảy ra. Vui lòng thử lại.",
            filled_slots=get_slots(user_id),
            missing_slots=[],
            next_action="error",
            mode=mode if 'mode' in locals() else "disease",
            debug={"error": str(e)}
        )

# ========= HELPER FUNCTIONS =========
def _extract_symptoms_from_text(text: str) -> List[str]:
    """
    Extract symptom names from Vietnamese text.
    This is a simplified version - in production you'd use NER or matching against symptom list.
    """
    # For now, return empty list - you'd implement symptom extraction here
    # This is a placeholder for more sophisticated NER
    symptoms = []

    # Simple keyword matching (you'd enhance this)
    text_lower = text.lower()
    for symptom in ann_service.symptom_columns:
        if symptom.lower() in text_lower:
            symptoms.append(symptom)

    return symptoms

def _parse_yes_no(text: str) -> Optional[bool]:
    """Parse yes/no answer from Vietnamese text"""
    text_lower = text.lower().strip()

    yes_words = ['có', 'co', 'yes', 'đúng', 'dung', 'uh', 'u', 'vâng', 'vang', 'phải', 'phai']
    no_words = ['không', 'khong', 'no', 'ko', 'k', 'không có', 'khong co']

    if any(word in text_lower for word in yes_words):
        return True
    elif any(word in text_lower for word in no_words):
        return False
    else:
        return None

def _format_diagnosis_result(prediction: PredictionResult, dept: str) -> str:
    """Format diagnosis result message"""
    return (
        f"🩺 **Kết quả phân tích:**\n"
        f"• Bệnh dự đoán: **{prediction.top_disease}** ({prediction.top_confidence*100:.1f}%)\n"
        f"• Các bệnh khác có thể: {', '.join([f'{d} ({p*100:.1f}%)' for d, p in prediction.top_k_predictions[1:4]])}\n"
        f"• Khoa khám gợi ý: **{dept}**\n\n"
        f"📊 Độ tin cậy: {prediction.confidence_category}\n\n"
        f"📋 Bạn có thể in hồ sơ tư vấn để mang đến bệnh viện.\n\n"
        f"❓ Bạn có muốn hỏi thêm điều gì khác không? Nếu không, hãy nhập 'kết thúc' để hoàn tất."
    )

def _update_patient_record(
    db: Session,
    user_id: str,
    filled: Dict,
    missing: List,
    prediction: Optional[PredictionResult],
    dept: Optional[str]
):
    """Update or create patient record in database"""
    try:
        existing_record = db.query(PatientRecord).filter(
            PatientRecord.user_id == user_id
        ).first()

        if existing_record:
            # Update existing
            existing_record.patient_name = filled.get("name", "")
            existing_record.patient_phone = filled.get("phone_number", "")
            existing_record.patient_age = filled.get("age", "")
            existing_record.patient_gender = filled.get("gender", "")
            existing_record.symptoms = filled.get("symptoms", "")
            existing_record.onset = filled.get("onset", "")
            existing_record.allergies = filled.get("allergies", "")
            existing_record.current_medications = filled.get("current_medications", "")
            existing_record.pain_scale = filled.get("pain_scale", "")

            if prediction:
                preds_dict = {d: float(p) for d, p in prediction.top_k_predictions}
                existing_record.predicted_diseases = json.dumps(preds_dict, ensure_ascii=False)
                existing_record.recommended_department = dept

            existing_record.chat = json.dumps(
                {"filled": filled, "missing": missing}, ensure_ascii=False
            )
        else:
            # Create new
            preds_dict = {}
            if prediction:
                preds_dict = {d: float(p) for d, p in prediction.top_k_predictions}

            record = PatientRecord(
                record_id=str(uuid.uuid4()),
                user_id=user_id,
                patient_name=filled.get("name", ""),
                patient_phone=filled.get("phone_number", ""),
                patient_age=filled.get("age", ""),
                patient_gender=filled.get("gender", ""),
                symptoms=filled.get("symptoms", ""),
                onset=filled.get("onset", ""),
                allergies=filled.get("allergies", ""),
                current_medications=filled.get("current_medications", ""),
                pain_scale=filled.get("pain_scale", ""),
                predicted_diseases=json.dumps(preds_dict, ensure_ascii=False),
                recommended_department=dept or DEPARTMENT_DEFAULT,
                chat=json.dumps({"filled": filled, "missing": missing}, ensure_ascii=False),
            )
            db.add(record)

        db.commit()
        logger.info(f"✅ Updated patient record for user {user_id}")

    except Exception as e:
        logger.error(f"Error updating patient record: {e}")
        db.rollback()

# ========= DATABASE ENDPOINTS (Same as V1) =========
@app.get("/patients")
def get_all_patients(db: Session = Depends(get_db)):
    """Get all patient records from database"""
    rows = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
    return [
        {
            "record_id": r.record_id,
            "user_id": r.user_id,
            "patient_name": r.patient_name,
            "patient_phone": r.patient_phone,
            "patient_age": r.patient_age,
            "patient_gender": r.patient_gender,
            "symptoms": r.symptoms,
            "onset": r.onset,
            "allergies": r.allergies,
            "current_medications": r.current_medications,
            "pain_scale": r.pain_scale,
            "predicted_diseases": json.loads(r.predicted_diseases) if r.predicted_diseases else {},
            "recommended_department": r.recommended_department,
            "created_at": r.created_at
        }
        for r in rows
    ]

@app.get("/statistics")
def get_statistics(db: Session = Depends(get_db)):
    """Get department statistics from patient records"""
    rows = db.query(PatientRecord.recommended_department).all()
    stats = {}
    for (dept,) in rows:
        dept = dept or "Unknown"
        stats[dept] = stats.get(dept, 0) + 1
    return stats

@app.get("/export_excel")
def export_excel(db: Session = Depends(get_db)):
    """Export patient records to Excel file"""
    try:
        rows = db.query(PatientRecord).order_by(PatientRecord.created_at.desc()).all()
        data = []

        for r in rows:
            data.append({
                "Record ID": r.record_id,
                "User ID": r.user_id,
                "Patient Name": r.patient_name or "",
                "Patient Phone": r.patient_phone or "",
                "Patient Age": r.patient_age or "",
                "Patient Gender": r.patient_gender or "",
                "Symptoms": r.symptoms or "",
                "Onset": r.onset or "",
                "Allergies": r.allergies or "",
                "Current Medications": r.current_medications or "",
                "Pain Scale": r.pain_scale or "",
                "Predicted Diseases": r.predicted_diseases or "",
                "Recommended Department": r.recommended_department or "",
                "Created At": str(r.created_at) if r.created_at else ""
            })

        df = pd.DataFrame(data)

        try:
            import openpyxl
            output_path = os.path.join(os.getcwd(), "patient_records_v2.xlsx")
            df.to_excel(output_path, index=False, engine='openpyxl')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "patient_records_v2.xlsx"
        except ImportError:
            output_path = os.path.join(os.getcwd(), "patient_records_v2.csv")
            df.to_csv(output_path, index=False, encoding='utf-8')
            media_type = "text/csv"
            filename = "patient_records_v2.csv"

        return FileResponse(
            output_path,
            filename=filename,
            media_type=media_type
        )

    except Exception as e:
        logger.error(f"Export error: {e}")
        return {"error": f"Failed to export: {str(e)}"}

# ========= MAIN =========
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Medical Chatbot V2 on {HOST}:{PORT}")
    logger.info(f"Model: PyTorch ANN with {ann_service.num_classes} diseases")
    logger.info(f"Active Questioning: Enabled with information gain calculation")
    uvicorn.run(app, host=HOST, port=PORT)
