#!/usr/bin/env python3
"""
Pre_2_ANN Project Structure Verification Script
================================================

This script verifies all file paths and imports are correct in the Pre_2_ANN project.

Usage:
    python verify_structure.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    if path.exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Missing: {path}")
        return False

def check_directory(path: Path, description: str) -> bool:
    """Check if a directory exists and print status."""
    if path.exists() and path.is_dir():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description}")
        print(f"   Missing: {path}")
        return False

def test_imports():
    """Test all critical imports."""
    print("\n" + "="*80)
    print("TESTING IMPORTS")
    print("="*80)

    tests = [
        ("Schemas", "from app_v2.schemas.chat_schemas import ChatIn, ChatOut, LLMOutput"),
        ("Database", "from app_v2.core.database import get_db"),
        ("Models", "from app_v2.models.database_models import PatientRecord"),
        ("State Store", "from app_v2.core.state_store import merge_slots"),
        ("Gemini Service", "from app_v2.services.gemini_service import call_gemini"),
        ("Active Questioning", "from app_v2.services.active_questioning import QuestioningSession"),
        ("ANN Service", "from app_v2.services.ann_service import ANNService"),
    ]

    all_passed = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"✅ {name:25s} - Import successful")
        except Exception as e:
            print(f"❌ {name:25s} - Error: {e}")
            all_passed = False

    return all_passed

def main():
    """Main verification function."""
    print("="*80)
    print("PRE_2_ANN PROJECT STRUCTURE VERIFICATION")
    print("="*80)
    print(f"Project Root: {PROJECT_ROOT}\n")

    all_checks = []

    # ========================================================================
    # 1. Configuration Files
    # ========================================================================
    print("="*80)
    print("1. CONFIGURATION FILES")
    print("="*80)

    all_checks.append(check_file_exists(
        PROJECT_ROOT / ".env",
        "Environment variables (.env)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / ".env.example",
        "Environment template (.env.example)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / ".gitignore",
        "Git ignore rules (.gitignore)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "requirements.txt",
        "Python dependencies (requirements.txt)"
    ))

    # ========================================================================
    # 2. Application Structure
    # ========================================================================
    print("\n" + "="*80)
    print("2. APPLICATION STRUCTURE (app_v2)")
    print("="*80)

    all_checks.append(check_directory(
        PROJECT_ROOT / "app_v2",
        "Application directory (app_v2/)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "main.py",
        "PhoBERT main file (app_v2/main.py)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "main_ann.py",
        "ANN main file (app_v2/main_ann.py) ⭐"
    ))

    # Core modules
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "core" / "database.py",
        "Database setup (app_v2/core/database.py)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "core" / "state_store.py",
        "State store (app_v2/core/state_store.py)"
    ))

    # Models
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "models" / "database_models.py",
        "Database models (app_v2/models/database_models.py)"
    ))

    # Schemas
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "schemas" / "chat_schemas.py",
        "Chat schemas (app_v2/schemas/chat_schemas.py)"
    ))

    # Services
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "services" / "ann_service.py",
        "ANN service (app_v2/services/ann_service.py) ⭐"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "services" / "active_questioning.py",
        "Active questioning (app_v2/services/active_questioning.py) ⭐"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "services" / "gemini_service.py",
        "Gemini service (app_v2/services/gemini_service.py)"
    ))

    # Utils
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "app_v2" / "utils" / "department_map.json",
        "Old department map (app_v2/utils/department_map.json)"
    ))

    # ========================================================================
    # 3. Building_ANN Structure
    # ========================================================================
    print("\n" + "="*80)
    print("3. MODEL TRAINING & DATA (building_ANN)")
    print("="*80)

    all_checks.append(check_directory(
        PROJECT_ROOT / "building_ANN",
        "Building directory (building_ANN/)"
    ))

    # Data
    all_checks.append(check_directory(
        PROJECT_ROOT / "building_ANN" / "data",
        "Data directory (building_ANN/data/)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "symptom_disease_dataset_7_converted.csv",
        "Raw dataset (building_ANN/data/symptom_disease_dataset_7_converted.csv)"
    ))

    # Processed data
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "processed" / "common" / "train.csv",
        "Training data (building_ANN/data/processed/common/train.csv)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "processed" / "common" / "val.csv",
        "Validation data (building_ANN/data/processed/common/val.csv)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "processed" / "common" / "test.csv",
        "Test data (building_ANN/data/processed/common/test.csv)"
    ))

    # Evaluation data
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "evaluation" / "most_confused_pairs.csv",
        "Confusion data (building_ANN/data/evaluation/most_confused_pairs.csv)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "data" / "evaluation" / "per_class_metrics.csv",
        "Per-class metrics (building_ANN/data/evaluation/per_class_metrics.csv)"
    ))

    # Models
    all_checks.append(check_directory(
        PROJECT_ROOT / "building_ANN" / "models" / "pytorch",
        "PyTorch models directory (building_ANN/models/pytorch/)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "models" / "pytorch" / "best_model.pth",
        "Best model (building_ANN/models/pytorch/best_model.pth) ⭐"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "models" / "pytorch" / "label_encoder.pkl",
        "Label encoder (building_ANN/models/pytorch/label_encoder.pkl) ⭐"
    ))

    # Utils
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "utils" / "department_map_577.json",
        "577-disease department map (building_ANN/utils/department_map_577.json) ⭐"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "utils" / "all_577_diseases.txt",
        "Disease list (building_ANN/utils/all_577_diseases.txt)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "utils" / "DEPARTMENT_MAPPING_README.md",
        "Department mapping docs (building_ANN/utils/DEPARTMENT_MAPPING_README.md)"
    ))

    # Notebooks
    all_checks.append(check_directory(
        PROJECT_ROOT / "building_ANN" / "notebooks",
        "Notebooks directory (building_ANN/notebooks/)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "notebooks" / "01_eda_and_label_preparation.ipynb",
        "EDA notebook (01_eda_and_label_preparation.ipynb)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "notebooks" / "02_data_preprocessing_common.ipynb",
        "Preprocessing notebook (02_data_preprocessing_common.ipynb)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "notebooks" / "03_train_ann_pytorch.ipynb",
        "Training notebook (03_train_ann_pytorch.ipynb)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "notebooks" / "04_model_evaluation.ipynb",
        "Evaluation notebook (04_model_evaluation.ipynb)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "notebooks" / "05_interactive_prediction_testing.ipynb",
        "Testing notebook (05_interactive_prediction_testing.ipynb)"
    ))

    # Python scripts
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "run_eda_analysis.py",
        "EDA script (building_ANN/run_eda_analysis.py)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "run_preprocessing_common_diseases.py",
        "Preprocessing script (building_ANN/run_preprocessing_common_diseases.py)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "building_ANN" / "run_training_pytorch.py",
        "Training script (building_ANN/run_training_pytorch.py)"
    ))

    # ========================================================================
    # 4. Documentation
    # ========================================================================
    print("\n" + "="*80)
    print("4. DOCUMENTATION")
    print("="*80)

    all_checks.append(check_file_exists(
        PROJECT_ROOT / "REFACTORING_SUMMARY.md",
        "Refactoring summary (REFACTORING_SUMMARY.md)"
    ))
    all_checks.append(check_file_exists(
        PROJECT_ROOT / "QUICK_START.md",
        "Quick start guide (QUICK_START.md)"
    ))

    # ========================================================================
    # 5. Verify No Duplicate Data Directory
    # ========================================================================
    print("\n" + "="*80)
    print("5. VERIFY NO DUPLICATE DIRECTORIES")
    print("="*80)

    notebooks_data = PROJECT_ROOT / "building_ANN" / "notebooks" / "data"
    if not notebooks_data.exists():
        print("✅ No duplicate data directory in notebooks/")
        all_checks.append(True)
    else:
        print("❌ Duplicate data directory found: building_ANN/notebooks/data/")
        print("   This should have been removed!")
        all_checks.append(False)

    # ========================================================================
    # 6. Test Imports
    # ========================================================================
    import_success = test_imports()
    all_checks.append(import_success)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    total = len(all_checks)
    passed = sum(all_checks)
    failed = total - passed

    print(f"\nTotal Checks: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\nSuccess Rate: {passed/total*100:.1f}%")

    if failed == 0:
        print("\n🎉 ALL CHECKS PASSED! Project structure is correct.")
        return 0
    else:
        print(f"\n⚠️  {failed} checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
