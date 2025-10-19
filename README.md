# Vietnamese Medical Diagnosis Chatbot System

An AI-powered medical diagnosis chatbot using Artificial Neural Networks (ANN) with active questioning capabilities for Vietnamese healthcare.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

This system provides intelligent preliminary medical diagnosis for Vietnamese patients using:
- **Deep Learning**: PyTorch ANN model with 577 disease classifications
- **Active Questioning**: Information theory-based intelligent symptom collection
- **High Accuracy**: 94.82% Top-3 accuracy, 83.22% Top-1 accuracy
- **Department Routing**: Automatic mapping to 24 hospital specialties
- **Conversational AI**: Natural language interaction powered by Gemini API

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| **Diseases Covered** | 577 |
| **Symptoms Tracked** | 377 |
| **Top-1 Accuracy** | 83.22% |
| **Top-3 Accuracy** | 94.82% |
| **Top-5 Accuracy** | 97.50% |
| **Hospital Departments** | 24 |
| **Average Questions** | 2.8 per session |
| **Confidence Improvement** | +51.78% average |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- MySQL 8.0+ (or SQLite for development)
- Gemini API key ([Get here](https://makersuite.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Pre_2_ANN.git
cd Pre_2_ANN

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Setup database (optional, for MySQL)
mysql -u root -p < setup_database.sql
```

### Running the Application

**Start Backend:**
```bash
python -m uvicorn app_v2.main_ann:app --host 0.0.0.0 --port 8004 --reload
```

**Start Frontend:**
```bash
cd frontend
python3 -m http.server 8080
```

**Access the Application:**
- Chatbot Interface: http://localhost:8080/index.html
- Admin Dashboard: http://localhost:8080/admin.html
- API Documentation: http://localhost:8004/docs

## 🏗️ Architecture

```
User Interface (HTML/JS)
         ↓
FastAPI Backend (Port 8004)
         ↓
    ┌────┴────┐
    ↓         ↓
ANN Model  Gemini API
(577 diseases) (Conversation)
    ↓
Active Questioning Engine
(Information Theory)
    ↓
Database (MySQL/SQLite)
```

## 📁 Project Structure

```
Pre_2_ANN/
├── app_v2/                      # FastAPI Application
│   ├── main_ann.py             # Main entry point
│   ├── services/
│   │   ├── ann_service.py      # ANN prediction engine
│   │   ├── active_questioning.py # Intelligent questioning
│   │   └── gemini_service.py   # Gemini API integration
│   ├── core/                   # Database & state management
│   ├── models/                 # Database ORM models
│   └── schemas/                # Pydantic schemas
│
├── building_ANN/               # Model Training & Data
│   ├── models/pytorch/         # Trained ANN model
│   ├── data/                   # Training/validation/test data
│   ├── notebooks/              # Jupyter notebooks (EDA, training, evaluation)
│   └── utils/                  # Department mapping & utilities
│
├── frontend/                   # Web Interface
│   ├── index.html             # Main chatbot UI
│   ├── app.js                 # Frontend logic
│   └── config.js              # Configuration
│
├── requirements.txt            # Python dependencies
├── setup_database.sql         # Database schema
└── CLAUDE.md                  # Development guide for AI assistants
```

## 🧠 Model Architecture

**ANN Design:**
```
Input Layer:    377 neurons (symptoms)
     ↓
Hidden Layer 1: 512 neurons + BatchNorm + ReLU + Dropout(0.3)
     ↓
Hidden Layer 2: 256 neurons + BatchNorm + ReLU + Dropout(0.3)
     ↓
Hidden Layer 3: 128 neurons + BatchNorm + ReLU + Dropout(0.2)
     ↓
Output Layer:   577 neurons (diseases) + Softmax
```

**Training Details:**
- Optimizer: Adam (lr=0.001)
- Loss: Cross Entropy with class weights
- Batch Size: 128
- Early Stopping: Patience = 10
- Device: MPS/CUDA/CPU compatible

## 🔍 Active Questioning System

The system uses a **hybrid questioning strategy** combining:

1. **Differential Diagnosis**: Distinguishes between top 2 diseases
2. **Information Gain**: Maximizes entropy reduction
3. **Entropy Reduction**: Broadens search for uncertain cases

**Stopping Criteria:**
- Confidence > 85% (HIGH)
- Top-2 gap > 40%
- Max 10 questions asked
- Information gain < 0.01

## 🏥 Department Mapping

Automatically routes patients to appropriate departments:

| Department | Examples |
|------------|----------|
| **Thần kinh** | Alzheimer, Parkinson, Epilepsy |
| **Tim mạch** | Heart disease, Hypertension |
| **Tiêu hoá** | Gastritis, Crohn's disease |
| **Nội hô hấp** | Pneumonia, Asthma, COPD |
| **Da liễu** | Dermatitis, Psoriasis |
| ...and 19 more |

## 📡 API Endpoints

### Main Endpoints

**POST /chat**
- Main conversational endpoint
- Collects patient information via dialogue
- Returns diagnosis with department recommendation

**POST /predict_symptoms**
- Direct disease prediction from symptom list
- No conversation, just prediction
- Useful for batch processing

**GET /health**
- Server health check
- Returns model status and statistics

See full API documentation at: http://localhost:8004/docs

## 🧪 Testing

Run the verification script:
```bash
python verify_structure.py
```

Expected output: All checks passed ✅

## 📊 Performance Metrics

**Model Evaluation:**
- **Test Accuracy**: 83.22% (Top-1), 94.82% (Top-3)
- **Validation Accuracy**: 83.75%
- **Training Accuracy**: 85.34%
- **Average Confidence**: 76.4% predictions >80% confidence

**Active Questioning:**
- **Average Questions**: 2.8 per session
- **Confidence Gain**: +51.78% average
- **Success Rate**: 96.4% reach >85% confidence

## 🛠️ Development

### Training the Model

```bash
cd building_ANN

# 1. Exploratory Data Analysis
python run_eda_analysis.py

# 2. Data Preprocessing
python run_preprocessing_common_diseases.py

# 3. Model Training
python run_training_pytorch.py
```

### Jupyter Notebooks

```bash
cd building_ANN/notebooks
jupyter notebook

# Available notebooks:
# 01_exploratory_data_analysis.ipynb
# 02_data_preprocessing.ipynb
# 03_ann_training_pytorch.ipynb
# 04_model_evaluation.ipynb
# 05_interactive_prediction_testing.ipynb
```

## 🔒 Security Notes

- Never commit `.env` file (contains API keys)
- Model files (`.pth`) are excluded from git due to size
- Database credentials should be changed in production
- CORS is configured for localhost by default

## 📚 Documentation

- **CLAUDE.md**: Comprehensive development guide for AI assistants
- **QUICK_START.md**: Quick reference for common tasks
- **ENV_CONFIGURATION.md**: Environment setup details
- **DATABASE_SETUP.md**: Database configuration guide
- **REFACTORING_SUMMARY.md**: Project structure and refactoring details

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Vietnamese medical diagnosis dataset (147,470 patient cases)
- AI Models: PyTorch, Google Gemini API
- Technologies: FastAPI, SQLAlchemy, Pandas, NumPy


