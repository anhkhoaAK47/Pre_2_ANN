# Vietnamese Medical Diagnosis Chatbot System

An AI-powered medical diagnosis chatbot using Artificial Neural Networks (ANN) with dual-mode active questioning for Vietnamese healthcare.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Status](https://img.shields.io/badge/Status-Fully%20Working-success.svg)]()

## Overview

This system provides intelligent preliminary medical diagnosis for Vietnamese patients with **two questioning modes**:

### **Disease-Based Mode** (Default)
- Direct disease prediction from 577 diseases
- Specific diagnosis with confidence scores
- Best for: Detailed medical analysis

### **Department-Based Mode** (Triage)
- Routes to 26 hospital departments
- Faster convergence through aggregation
- Best for: Hospital triage and referrals

### Key Features
- **Deep Learning**: PyTorch ANN model trained on 147,470 patient cases
- **Dual Questioning Modes**: Disease-specific OR department-based
- **High Accuracy**: 94.82% Top-3 accuracy, 83.22% Top-1 accuracy
- **Intelligent Questioning**: Information theory-based symptom collection
- **Natural Conversation**: Powered by Google Gemini 2.0 Flash
- **Full Vietnamese Support**: Language and medical terminology

## Performance Metrics

| Metric | Value |
|--------|-------|
| **Diseases Covered** | 577 |
| **Hospital Departments** | 26 |
| **Symptoms Tracked** | 377 |
| **Top-1 Accuracy** | 83.22% |
| **Top-3 Accuracy** | 94.82% |
| **Top-5 Accuracy** | 97.50% |
| **Avg Questions (Disease)** | 1-3 questions |
| **Avg Questions (Department)** | 1-2 questions |
| **Confidence Gain** | +42-52% average |

## Quick Start

### Prerequisites

- Python 3.10+
- MySQL 8.0+ or SQLite
- Gemini API key → [Get here](https://makersuite.google.com/app/apikey)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Pre_2_ANN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Accessing Models and Data

Due to GitHub's file size limitations, the following resources are not included in this repository:

- **Trained Models** (`building_ANN/models/pytorch/`)
  - `best_model.pth` (~450K parameters, ~40MB)
  - `label_encoder.pkl` (disease label mappings)

- **Training Data** (`building_ANN/data/`)
  - `symptom_disease_dataset_7_converted.csv` (~192MB, 147,470 cases)
  - Processed train/val/test splits

- **Notebooks** (`notebooks/`, `building_ANN/notebooks/`)
  - Analysis and training notebooks

**To request access to models and datasets:**
📧 Contact: **quanghini@gmail.com**

### Running the System

**1. Start Backend:**
```bash
python -m uvicorn app_v2.main_ann:app --host 0.0.0.0 --port 8004 --reload
```

**2. Start Frontend:**
```bash
cd frontend
python3 -m http.server 3000
```

**3. Access Application:**
- **Chatbot UI**: http://localhost:3000/index.html
- **Dashboard**: http://localhost:3000/admin.html
- **API Docs**: http://localhost:8004/docs

### Stop Servers

```bash
# Backend
lsof -ti:8004 | xargs kill -9

# Frontend
Ctrl + C
```

## Architecture

```
┌─────────────────────────────────────────────┐
│          Frontend (HTML/JS)                 │
│     Chatbot UI    |    Admin Dashboard     │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────┐
│      FastAPI Backend (Port 8004)             │
│                                              │
│  ┌──────────────┐      ┌─────────────────┐  │
│  │ Gemini API   │      │   ANN Service   │  │
│  │ (Gemini 2.0) │      │  (577 diseases) │  │
│  └──────┬───────┘      └────────┬────────┘  │
│         │                       │           │
│         ↓                       ↓           │
│  ┌──────────────────────────────────────┐   │
│  │   Active Questioning Engine          │   │
│  │   • Disease Mode (577 targets)       │   │
│  │   • Department Mode (26 targets)     │   │
│  │   • Hybrid strategy (Info gain +     │   │
│  │     Differential + Entropy)          │   │
│  └──────────────────┬───────────────────┘   │
└────────────────────┬────────────────────────┘
                     │
                     ↓
          ┌──────────────────────┐
          │ Database             │
          │ (MySQL / SQLite)     │
          │ • Patient records    │
          │ • Questioning logs   │
          │ • Model metrics      │
          └──────────────────────┘
```

## Dual Questioning Modes

### Disease-Based Questioning

**How it works:**
1. Predicts from 577 specific diseases
2. Asks questions to distinguish similar diseases
3. Uses confusion matrix to identify commonly confused pairs
4. Returns specific disease diagnosis

**Example:**
```
Input: "đau đầu, sốt cao"
Initial: 39.3% confidence (ngộ độc khí carbon monoxide)
Question 1: "Bạn có triệu chứng trầm cảm hoặc loạn thần không?"
→ Answer: "Có"
Result: 98.9% confidence 
Output: ngộ độc khí carbon monoxide → Khám tổng quát
```

### Department-Based Questioning

**How it works:**
1. Aggregates disease probabilities by department (SUM method)
2. Asks questions to distinguish between departments
3. Faster convergence (26 targets vs 577)
4. Returns department + top disease

**Example:**
```
Input: "đau đầu, buồn nôn"
Initial: 57.0% confidence (Khám tổng quát)
Question 1: "Bạn có đau bụng dữ dội không?"
→ Answer: "Có"
Question 2: "Bạn có táo bón không?"
→ Answer: "Có"
Result: 99.6% confidence 
Output: Ngoại tổng quát / Cấp cứu → tắc ruột
```

### When to Use Each Mode

| Use Case | Recommended Mode |
|----------|------------------|
| Hospital triage | **Department** |
| Primary care routing | **Department** |
| Specific diagnosis research | **Disease** |
| Medical education | **Disease** |
| Telemedicine consultation | **Disease** |

## 📁 Project Structure

```
Pre_2_ANN/
├── app_v2/                          # FastAPI Application
│   ├── main_ann.py                 # Main entry (FULLY WORKING)
│   ├── services/
│   │   ├── ann_service.py          # ANN prediction engine
│   │   ├── active_questioning.py   # Disease-based questioning
│   │   ├── department_questioning.py # Department-based questioning
│   │   └── gemini_service.py       # Gemini 2.0 Flash integration
│   ├── core/                       # Database & state
│   ├── models/                     # ORM models
│   └── schemas/
│       └── chat_schemas.py         # Updated with active_questioning_response
│
├── building_ANN/                    # Model Training & Research
│   ├── models/pytorch/
│   │   ├── best_model.pth          # Trained ANN (577 diseases)
│   │   └── label_encoder.pkl       # Disease encoder
│   ├── data/
│   │   ├── processed/common/       # Train/Val/Test splits
│   │   └── evaluation/             # Confusion matrices
│   ├── notebooks/
│   │   ├── 06_department_based_active_questioning.ipynb
│   │   └── 07_disease_based_active_questioning.ipynb
│   └── utils/
│       └── department_map_577.json # Disease → Department mapping
│
├── frontend/                        # Web Interface
│   ├── index.html                  # Chatbot UI
│   ├── admin.html                  # Dashboard
│   ├── app.js                      # Frontend logic
│   └── config.js                   # API configuration
│
└── docs/                           # Documentation
    ├── CLAUDE.md                   # AI assistant development guide
    └── resume.md                   # Project summary
```

## Model Architecture

**ANN Design** (PyTorch):
```
Input:        377 symptoms (binary vector)
              ↓
Layer 1:      377 → 512 + BatchNorm + ReLU + Dropout(0.3)
              ↓
Layer 2:      512 → 256 + BatchNorm + ReLU + Dropout(0.3)
              ↓
Layer 3:      256 → 128 + BatchNorm + ReLU + Dropout(0.2)
              ↓
Output:       128 → 577 diseases + Softmax
```

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class weights
- Batch Size: 128
- Early Stopping: Patience = 10
- Dataset: 147,470 cases → 70/15/15 split

## API Usage

### Chat Endpoint (Main)

```python
import requests

# Disease mode
response = requests.post("http://localhost:8004/chat", json={
    "user_id": "user123",
    "message": "Tôi bị đau đầu và sốt",
    "mode": "disease"  # or "department"
})

print(response.json())
```

**Response:**
```json
{
  "assistant_message": "Bạn có thể cho mình biết tên đầy đủ...",
  "filled_slots": {"symptoms": "đau đầu, sốt"},
  "missing_slots": ["name", "phone_number", ...],
  "next_action": "ask_for_missing_slots",
  "mode": "disease"
}
```

### Direct Prediction

```python
# Skip conversation, direct prediction
response = requests.post("http://localhost:8004/predict_symptoms", json={
    "symptoms": ["đau đầu", "sốt cao", "buồn nôn"],
    "top_k": 5
})
```

## Active Questioning Algorithm

**Hybrid Strategy** (combines 3 approaches):

1. **Differential Diagnosis**
   - Targets confused disease pairs from confusion matrix
   - Asks discriminating questions between top 2 candidates

2. **Information Gain**
   - Calculates H(Y) - E[H(Y|X)] for each symptom
   - Selects symptom that maximally reduces uncertainty

3. **Entropy Reduction**
   - Minimizes expected entropy across all hypotheses
   - Broadens search when top candidates are unclear

**Stopping Criteria:**
-  Top disease probability > 85%
-  Gap between #1 and #2 > 40%
-  10 questions asked (max)
-  Information gain < 0.01 (diminishing returns)

## Department Mapping

**26 Hospital Departments:**

| Vietnamese | English | Example Diseases |
|------------|---------|------------------|
| Thần kinh | Neurology | Alzheimer, Parkinson, Epilepsy |
| Tim mạch | Cardiology | Heart disease, Hypertension |
| Tiêu hoá | Gastroenterology | Gastritis, Crohn's, IBS |
| Nội hô hấp | Pulmonology | Pneumonia, Asthma, COPD |
| Nội tiết | Endocrinology | Diabetes, Thyroid |
| Da liễu | Dermatology | Psoriasis, Eczema |
| Sản phụ khoa | OB/GYN | Pregnancy, Menstrual |
| Nam khoa | Urology | Prostate, UTI |
| Cơ xương khớp | Orthopedics | Arthritis, Fractures |
| ... | ... | ... (17 more) |

**Mapping Coverage:**
- Specific mappings: 40+ diseases
- Pattern matching: 340+ diseases
- Fallback: "Khám tổng quát" (General)
- Total coverage: 65.7% automatic, 34.3% default

## Testing

```bash
# Test imports
python3 -c "from app_v2.services.ann_service import ANNService; print(' OK')"

# Test API
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Tôi bị đau đầu"}'

# Run notebooks (optional)
cd building_ANN/notebooks
jupyter notebook
# Open 06 (department) or 07 (disease) for testing
```

## Database Schema

**patient_records:**
- Patient info: name, phone, age, gender
- Medical: symptoms (JSON), allergies, medications, pain_scale
- Diagnosis: predicted_diseases (JSON), top_disease, confidence
- Questioning: questions_asked, strategy, confidence_gain
- Department: recommended_department

**questioning_sessions:**
- Tracks active sessions
- Logs questions and responses
- Links to patient_records (FK)

**model_metrics:**
- Daily performance monitoring
- Aggregated statistics
- Department/disease distributions

## Development

### Training New Models

```bash
cd building_ANN

# 1. Data analysis
python run_eda_analysis.py

# 2. Preprocessing
python run_preprocessing_common_diseases.py

# 3. Training
python run_training_pytorch.py

# Model saved to: building_ANN/models/pytorch/best_model.pth
```

### Notebooks for Research

```bash
cd building_ANN/notebooks
jupyter notebook

# Available:
# 01 - Exploratory Data Analysis
# 02 - Data Preprocessing
# 03 - ANN Training (PyTorch)
# 04 - Model Evaluation
# 05 - Interactive Testing
# 06 - Department-Based Questioning 
# 07 - Disease-Based Questioning 
# 08 - Comparison Analysis
```

## Configuration

**Environment Variables** (.env):
```bash
# Required
GEMINI_API_KEY=your_api_key_here
DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/medical_chatbot_ann

# Optional (auto-configured)
HOST=0.0.0.0
PORT=8004
GEMINI_MODEL=gemini-2.0-flash
DEPARTMENT_DEFAULT=Khám tổng quát
```

**Model Paths** (auto-configured in main_ann.py):
- MODEL_PATH: `building_ANN/models/pytorch/best_model.pth`
- LABEL_ENCODER: `building_ANN/models/pytorch/label_encoder.pkl`
- TRAIN_DATA: `building_ANN/data/processed/common/train.csv`
- DEPT_MAP: `building_ANN/utils/department_map_577.json`


## Troubleshooting

**Import errors:**
```bash
# Use app_v2 prefix (not app)
from app_v2.services.ann_service import ANNService
```

**Port already in use:**
```bash
lsof -ti:8004 | xargs kill -9
```

**Model loading errors:**
```bash
# Ensure running from project root
cd /path/to/Pre_2_ANN
python -m uvicorn app_v2.main_ann:app --port 8004
```
it push origin feature/NewFeature`)
5. Open Pull Request

.

## Acknowledgments

- **Dataset**: Vietnamese medical diagnosis dataset (147,470 patient cases)
- **AI Models**: PyTorch, Google Gemini 2.0 Flash
- **Framework**: FastAPI, SQLAlchemy
- **Libraries**: Pandas, NumPy, Scikit-learn


