# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Vietnamese medical diagnosis chatbot system using an Artificial Neural Network (ANN) with active questioning capabilities. The system predicts diseases from symptoms with 94.82% Top-3 accuracy across 577 diseases, using intelligent information theory-based questioning to improve diagnostic confidence.

**Key Stats:**
- **577 diseases** mapped to 24 hospital departments
- **377 symptoms** in the feature space
- **83.22% Top-1 accuracy, 94.82% Top-3 accuracy**
- **Average 2.8 questions** to reach high confidence (>85%)

## Common Commands

### Running the Application

```bash
# Start ANN-based chatbot (recommended, port 8004)
cd app_v2
python -m uvicorn main_ann:app --host 0.0.0.0 --port 8004 --reload

# Start legacy PhoBERT-based chatbot (port 8003)
cd app_v2
python -m uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### Model Training & Data Processing

```bash
# Run exploratory data analysis
cd building_ANN
python run_eda_analysis.py

# Preprocess data and create train/val/test splits
python run_preprocessing_common_diseases.py

# Train the ANN model
python run_training_pytorch.py
```

### Testing

```bash
# Test imports
cd /path/to/Pre_2_ANN
python3 -c "from app_v2.services.ann_service import ANNService; print('✅ Imports OK')"

# Test API endpoint
curl -X POST http://localhost:8004/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test123", "message": "Tôi bị đau đầu và sốt cao"}'

# Verify structure
python verify_structure.py
```

### Database Setup

```bash
# MySQL setup (production)
mysql -u root -p < setup_database.sql

# SQLite is used automatically for development if DATABASE_URL uses sqlite://
```

### Jupyter Notebooks

```bash
cd building_ANN/notebooks
jupyter notebook

# Available notebooks (in order):
# 01_exploratory_data_analysis.ipynb - EDA & data exploration
# 02_data_preprocessing.ipynb - Data cleaning & splitting
# 03_ann_training_pytorch.ipynb - Model training
# 04_model_evaluation.ipynb - Performance evaluation
# 05_interactive_prediction_testing.ipynb - Interactive testing
```

## Architecture & Code Structure

### Two-Tier System Architecture

```
User → FastAPI App → ANN Prediction Engine → Active Questioning → Database
                  ↓
             Gemini API (conversation management)
```

**app_v2/**: FastAPI application with two versions
- `main_ann.py` (⭐ recommended): ANN-based prediction with active questioning
- `main.py` (legacy): PhoBERT-based prediction (57 diseases only)

**building_ANN/**: Model training, data, and evaluation
- `data/`: Raw and processed datasets
- `models/pytorch/`: Trained ANN model (best_model.pth) and label encoder
- `notebooks/`: Jupyter notebooks for development workflow
- `utils/`: Department mapping and utilities

### Key Service Modules

**app_v2/services/ann_service.py** (~550 lines)
- `ANNDiseaseClassifier`: PyTorch model (377 → 512 → 256 → 128 → 577)
- `ANNService`: Main prediction service with model loading, symptom extraction, and prediction
- Returns `PredictionResult` with confidence categorization (HIGH/MEDIUM/LOW)
- Integrates with active questioning engine

**app_v2/services/active_questioning.py** (~650 lines)
- `ActiveQuestioningEngine`: Information theory-based questioning
- Three strategies: DIFFERENTIAL (confused pairs), INFORMATION_GAIN, ENTROPY
- Default: HYBRID strategy combining all three
- Bayesian probability updating after each answer
- Stopping criteria: >85% confidence, >40% probability gap, max 10 questions, or low info gain

**app_v2/services/gemini_service.py** (~700 lines)
- `call_gemini()`: Manages conversation flow with Gemini API
- Extracts symptoms from Vietnamese natural language
- Handles slot filling for patient information
- Integrates with ANNService for predictions

### Critical File Paths (Auto-configured)

All paths are automatically resolved relative to `Pre_2_ANN/` in `main_ann.py`:

```python
MODEL_PATH = "building_ANN/models/pytorch/best_model.pth"
LABEL_ENCODER = "building_ANN/models/pytorch/label_encoder.pkl"
TRAIN_DATA = "building_ANN/data/processed/common/train.csv"
CONFUSION_MATRIX = "building_ANN/data/evaluation/most_confused_pairs.csv"
DEPARTMENT_MAP = "building_ANN/utils/department_map_577.json"  # 577 diseases → 24 departments
```

**Do not hardcode absolute paths** - the code uses `BASE_DIR` to resolve paths dynamically.

## Important Implementation Details

### Import Pattern

Always use `app_v2.` prefix for imports within the application:

```python
# Correct
from app_v2.core.database import get_db
from app_v2.services.ann_service import ANNService
from app_v2.schemas.chat_schemas import ChatIn, ChatOut

# Incorrect - will fail
from app.core.database import get_db  # Old pattern
```

### Department Mapping System

The system uses a **577-disease mapping** in `building_ANN/utils/department_map_577.json`:
- **Hybrid approach**: Specific mappings (40+) + pattern matching (regex) + legacy (57) + fallback
- **65.7% automatic coverage**, 34.3% use default "Khám tổng quát"
- See `building_ANN/utils/DEPARTMENT_MAPPING_README.md` for details

When adding new disease mappings:
1. Edit `building_ANN/utils/create_department_mapping_v2.py`
2. Add to `SPECIFIC_MAPPINGS` or `DEPARTMENT_PATTERNS`
3. Regenerate: `python create_department_mapping_v2.py`

### Active Questioning Flow

The system triggers active questioning when:
- **MEDIUM confidence** (60-85%): 1-5 questions
- **LOW confidence** (<60%): 3-10 questions

Questioning stops when:
- Top disease probability > 85% (HIGH confidence)
- Gap between #1 and #2 > 40%
- 10 questions reached (max)
- Information gain < 0.01 (diminishing returns)

### Confidence Categories

```python
# In ann_service.py
if confidence >= 0.85:
    return "HIGH"    # Direct recommendation
elif confidence >= 0.60:
    return "MEDIUM"  # Light questioning (1-3 questions)
else:
    return "LOW"     # Heavy questioning (3-10 questions)
```

### Data Splits

Training data is in `building_ANN/data/processed/common/`:
- **train.csv**: 132,188 samples (70%) - stratified split
- **val.csv**: 28,326 samples (15%)
- **test.csv**: 28,326 samples (15%)

Original dataset: `building_ANN/data/symptom_disease_dataset_7_converted.csv` (192 MB, 147,470 unique cases)

### Model Architecture Details

```
Input: 377 binary features (symptoms)
  ↓
Layer 1: Linear(377, 512) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Layer 2: Linear(512, 256) → BatchNorm → ReLU → Dropout(0.3)
  ↓
Layer 3: Linear(256, 128) → BatchNorm → ReLU → Dropout(0.2)
  ↓
Output: Linear(128, 577) → Softmax
```

**Training configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss with class weights
- Batch size: 128
- Early stopping: patience=10
- Total parameters: ~450,000

## Environment Configuration

Required environment variables in `.env`:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here  # Get from https://makersuite.google.com/app/apikey
DATABASE_URL=sqlite:///./medical_chatbot.db  # Or mysql+pymysql://user:pass@host:port/db

# Optional (good defaults)
HOST=0.0.0.0
PORT=8004
GEMINI_MODEL=gemini-2.0-flash
DEPARTMENT_DEFAULT=Khám tổng quát
```

**Note**: Model paths (MODEL_PATH, LABEL_ENCODER_PATH, etc.) should NOT be set in `.env` - they are auto-configured.

## Database Schema

**patient_records** table:
- Patient info: name, phone, age, gender
- Medical data: symptoms (JSON), allergies, medications, pain_scale
- Diagnosis: predicted_diseases (JSON), top_disease, top_confidence, confidence_category
- Questioning: questions_asked, questioning_strategy, initial_confidence, final_confidence
- Department: recommended_department
- Metadata: chat (JSON), created_at

**questioning_sessions** table:
- Tracks active questioning sessions
- Stores questions_log (JSON) and hypotheses_history (JSON)
- Links to patient_records via record_id (FK)

**model_metrics** table:
- Daily performance monitoring
- Aggregated statistics: avg_confidence, avg_questions_asked, confidence_gain
- Distribution data: department_distribution (JSON), top_diseases (JSON)

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'app'"
- Fix: Change imports from `app.` to `app_v2.`
- The old codebase used `app/`, this repo uses `app_v2/`

### "FileNotFoundError: [model file]"
- Fix: Ensure you're running from `Pre_2_ANN/` directory or `app_v2/` subdirectory
- Paths are resolved relative to `BASE_DIR` (project root)
- Check files exist: `ls building_ANN/models/pytorch/`

### "ModuleNotFoundError: No module named 'numpy._core'"
- This indicates label_encoder.pkl was pickled with incompatible numpy version
- Solution: Recreate label encoder with current environment
- Run: `python building_ANN/run_training_pytorch.py` (regenerates encoder)

### Port Already in Use
- Check if server is already running: `lsof -ti:8004`
- Kill existing process: `lsof -ti:8004 | xargs kill -9`
- Or change PORT in `.env`

### Missing GEMINI_API_KEY
- Error: "GEMINI_API_KEY chưa được đặt trong .env"
- Get API key from: https://makersuite.google.com/app/apikey
- Add to `.env`: `GEMINI_API_KEY=your_key_here`

## Development Workflow

### Making Changes to the Model

1. **Data changes**: Edit `building_ANN/run_preprocessing_common_diseases.py`
2. **Architecture changes**: Edit `building_ANN/run_training_pytorch.py` (modify `ANNDiseaseClassifier`)
3. **Train**: Run `python building_ANN/run_training_pytorch.py`
4. **Evaluate**: Use `building_ANN/notebooks/04_model_evaluation.ipynb`
5. **Deploy**: Model automatically saved to `building_ANN/models/pytorch/best_model.pth`

### Making Changes to Active Questioning

1. **Algorithm changes**: Edit `app_v2/services/active_questioning.py`
2. **Strategy changes**: Modify `QuestioningStrategy` enum or `ActiveQuestioningEngine` class
3. **Thresholds**: Adjust in `main_ann.py` when initializing `ActiveQuestioningEngine`:
   - `max_questions` (default: 10)
   - `confidence_threshold` (default: 0.85)
   - `gap_threshold` (default: 0.40)

### Making Changes to API Endpoints

1. **Edit**: `app_v2/main_ann.py` (for ANN version)
2. **Schemas**: Update `app_v2/schemas/chat_schemas.py` for request/response models
3. **Test**: Restart server with `--reload` flag to auto-reload on changes
4. **Verify**: Use curl or Postman to test endpoints

## Testing Strategy

**Unit tests** should cover:
- Model loading and initialization
- Symptom extraction from Vietnamese text
- Disease prediction with known symptom vectors
- Confidence categorization
- Active questioning trigger logic
- Question selection algorithms
- Department mapping accuracy

**Integration tests** should cover:
- End-to-end chatbot flow
- Database persistence
- Gemini API integration
- FastAPI endpoint responses

See `TEST_REPORT.md` for current test coverage (100% as of 2025-10-19).

## Documentation

- `FINAL_PROJECT_REPORT.md`: Complete project documentation with performance metrics
- `REFACTORING_SUMMARY.md`: Details on code structure and refactoring
- `QUICK_START.md`: Quick reference for common tasks
- `ENV_CONFIGURATION.md`: Detailed environment setup guide
- `DATABASE_SETUP.md`: Database schema and setup instructions
- `building_ANN/utils/DEPARTMENT_MAPPING_README.md`: Department mapping documentation

## Language & Localization

- **Primary language**: Vietnamese (symptoms, diseases, questions)
- **Symptoms**: 377 Vietnamese symptom names (e.g., "đau đầu", "sốt cao")
- **Diseases**: 577 Vietnamese disease names (e.g., "bệnh Alzheimer", "viêm phổi")
- **Departments**: 24 Vietnamese department names (e.g., "Tim mạch", "Thần kinh")

When adding new symptoms or diseases, ensure Vietnamese text is properly encoded (UTF-8).

## Performance Considerations

- **Inference time**: <100ms per prediction (CPU)
- **Question selection**: <50ms per question (cached disease-symptom matrix)
- **Memory**: ~200MB for model + data in memory
- **Database**: Use connection pooling for production (configured in `app_v2/core/database.py`)

## Production Deployment Notes

1. **Use MySQL** instead of SQLite for production reliability
2. **Set appropriate** `DATABASE_URL` with production credentials
3. **Use Gunicorn** or similar WSGI server (not uvicorn --reload)
4. **Enable CORS** if serving from different domain (configured in `main_ann.py`)
5. **Monitor** using `model_metrics` table for daily performance tracking
6. **Rate limit** Gemini API calls to avoid quota exhaustion
7. **Log** all predictions and questioning sessions for analytics
