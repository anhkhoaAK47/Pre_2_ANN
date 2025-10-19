#!/usr/bin/env python3
"""
Preprocessing Script for Common Diseases Only (577 diseases)
=============================================================

This script preprocesses ONLY the 577 common diseases (>10 samples each)
to ensure reliable stratified splitting and model evaluation.

The 191 rare diseases (≤10 samples) will be added later when more data is available.

Usage:
    python run_preprocessing_common_diseases.py

Outputs:
    - data/processed/common/train.csv
    - data/processed/common/val.csv
    - data/processed/common/test.csv
    - data/processed/common/class_weights.json
    - data/processed/common/preprocessing_report.json
    - Visualization charts
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

np.random.seed(RANDOM_SEED)

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path('../data/processed')
OUTPUT_DIR = DATA_DIR / 'common'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("PREPROCESSING: COMMON DISEASES ONLY (577 diseases)")
    print("="*80)
    print(f"Strategy: Train only on diseases with >10 samples")
    print(f"Rare diseases (191) will be added later with more data")
    print()

    # -------------------------------------------------------------------------
    # 1. LOAD COMMON DISEASES DATA
    # -------------------------------------------------------------------------
    print("📂 Step 1: Loading common diseases data...")

    common_data = pd.read_csv(DATA_DIR / 'common_diseases_data.csv')

    print(f"   Total samples: {len(common_data):,}")
    print(f"   Total features: {len(common_data.columns) - 1}")
    print(f"   Total diseases: {common_data['disease'].nunique()}")

    # Verify all diseases have >10 samples
    disease_counts = common_data['disease'].value_counts()
    min_samples = disease_counts.min()
    max_samples = disease_counts.max()

    print(f"\n   Sample distribution:")
    print(f"   Min samples per disease: {min_samples}")
    print(f"   Max samples per disease: {max_samples}")
    print(f"   Mean samples per disease: {disease_counts.mean():.1f}")
    print(f"   Median samples per disease: {disease_counts.median():.1f}")
    print()

    # -------------------------------------------------------------------------
    # 2. PREPARE FEATURES AND LABELS
    # -------------------------------------------------------------------------
    print("🔧 Step 2: Preparing features and labels...")

    X = common_data.drop('disease', axis=1)
    y = common_data['disease']

    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print()

    # -------------------------------------------------------------------------
    # 3. STRATIFIED TRAIN/VAL/TEST SPLIT
    # -------------------------------------------------------------------------
    print("✂️  Step 3: Stratified train/val/test split...")
    print(f"   Split ratios: {TRAIN_RATIO:.0%} / {VAL_RATIO:.0%} / {TEST_RATIO:.0%}")

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: separate train and validation
    val_ratio_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_of_temp,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    print(f"\n   ✅ Split complete:")
    print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(common_data)*100:.2f}%)")
    print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(common_data)*100:.2f}%)")
    print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(common_data)*100:.2f}%)")
    print(f"   Total: {len(X_train) + len(X_val) + len(X_test):,} samples")
    print()

    # -------------------------------------------------------------------------
    # 4. VERIFY STRATIFICATION
    # -------------------------------------------------------------------------
    print("🔍 Step 4: Verifying stratification...")

    train_disease_counts = y_train.value_counts()
    val_disease_counts = y_val.value_counts()
    test_disease_counts = y_test.value_counts()

    print(f"   Diseases in train: {len(train_disease_counts)}")
    print(f"   Diseases in val:   {len(val_disease_counts)}")
    print(f"   Diseases in test:  {len(test_disease_counts)}")

    # Check if all diseases are in all splits
    all_in_all_splits = (
        len(train_disease_counts) == len(val_disease_counts) ==
        len(test_disease_counts) == disease_counts.nunique()
    )

    if all_in_all_splits:
        print(f"   ✅ All {disease_counts.nunique()} diseases present in all splits!")
    else:
        print(f"   ⚠️  Warning: Not all diseases in all splits")
        print(f"       Missing from val: {disease_counts.nunique() - len(val_disease_counts)}")
        print(f"       Missing from test: {disease_counts.nunique() - len(test_disease_counts)}")

    # Calculate stratification quality
    max_deviation = 0
    for disease in disease_counts.index:
        original_ratio = disease_counts[disease] / len(common_data)
        train_ratio = train_disease_counts.get(disease, 0) / len(y_train)
        deviation = abs(original_ratio - train_ratio)
        max_deviation = max(max_deviation, deviation)

    print(f"   Max stratification deviation: {max_deviation*100:.3f}%")
    print()

    # -------------------------------------------------------------------------
    # 5. COMPUTE CLASS WEIGHTS
    # -------------------------------------------------------------------------
    print("⚖️  Step 5: Computing class weights...")

    unique_classes = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )

    class_weights_dict = {
        disease: float(weight)
        for disease, weight in zip(unique_classes, class_weights_array)
    }

    weights_array = np.array(list(class_weights_dict.values()))
    print(f"   Class weights computed: {len(class_weights_dict)} diseases")
    print(f"   Weight range: {weights_array.min():.3f} to {weights_array.max():.3f}")
    print(f"   Weight ratio: {weights_array.max() / weights_array.min():.1f}:1")
    print()

    # -------------------------------------------------------------------------
    # 6. SAVE DATASETS
    # -------------------------------------------------------------------------
    print("💾 Step 6: Saving datasets...")

    # Reconstruct dataframes
    train_df = pd.concat([y_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    val_df = pd.concat([y_val.reset_index(drop=True), X_val.reset_index(drop=True)], axis=1)
    test_df = pd.concat([y_test.reset_index(drop=True), X_test.reset_index(drop=True)], axis=1)

    # Save CSV files
    train_path = OUTPUT_DIR / 'train.csv'
    val_path = OUTPUT_DIR / 'val.csv'
    test_path = OUTPUT_DIR / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   ✅ {train_path} ({len(train_df):,} samples)")
    print(f"   ✅ {val_path} ({len(val_df):,} samples)")
    print(f"   ✅ {test_path} ({len(test_df):,} samples)")

    # Save class weights
    weights_path = OUTPUT_DIR / 'class_weights.json'
    with open(weights_path, 'w', encoding='utf-8') as f:
        json.dump(class_weights_dict, f, indent=2, ensure_ascii=False)
    print(f"   ✅ {weights_path}")
    print()

    # -------------------------------------------------------------------------
    # 7. CREATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("📊 Step 7: Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Split sizes
    split_sizes = [len(train_df), len(val_df), len(test_df)]
    split_labels = ['Train', 'Validation', 'Test']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    axes[0, 0].bar(split_labels, split_sizes, color=colors)
    axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
    axes[0, 0].set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(split_sizes):
        axes[0, 0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

    # Plot 2: Disease distribution across splits
    split_disease_counts = [len(train_disease_counts), len(val_disease_counts), len(test_disease_counts)]
    axes[0, 1].bar(split_labels, split_disease_counts, color=colors)
    axes[0, 1].set_ylabel('Number of Diseases', fontsize=12)
    axes[0, 1].set_title('Diseases per Split', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(split_disease_counts):
        axes[0, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

    # Plot 3: Top 20 diseases
    top20 = disease_counts.head(20)
    axes[1, 0].barh(range(len(top20)), top20.values, color='#3498db')
    axes[1, 0].set_yticks(range(len(top20)))
    axes[1, 0].set_yticklabels(top20.index, fontsize=9)
    axes[1, 0].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 0].set_title('Top 20 Most Common Diseases', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Plot 4: Sample distribution histogram
    axes[1, 1].hist(disease_counts.values, bins=50, color='#e74c3c', edgecolor='black')
    axes[1, 1].set_xlabel('Samples per Disease', fontsize=12)
    axes[1, 1].set_ylabel('Number of Diseases', fontsize=12)
    axes[1, 1].set_title('Sample Distribution (577 diseases)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(disease_counts.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: {disease_counts.mean():.1f}')
    axes[1, 1].axvline(disease_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {disease_counts.median():.1f}')
    axes[1, 1].legend()

    plt.tight_layout()
    viz_path = OUTPUT_DIR / 'preprocessing_visualizations.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ {viz_path}")
    print()

    # -------------------------------------------------------------------------
    # 8. GENERATE REPORT
    # -------------------------------------------------------------------------
    print("📋 Step 8: Generating preprocessing report...")

    report = {
        'preprocessing_info': {
            'strategy': 'Common diseases only (>10 samples)',
            'total_diseases': int(disease_counts.nunique()),
            'rare_diseases_excluded': 191,
            'random_seed': RANDOM_SEED,
            'split_ratios': {
                'train': TRAIN_RATIO,
                'val': VAL_RATIO,
                'test': TEST_RATIO
            }
        },
        'dataset_statistics': {
            'total_samples': int(len(common_data)),
            'total_features': int(len(X.columns)),
            'train_samples': int(len(train_df)),
            'val_samples': int(len(val_df)),
            'test_samples': int(len(test_df))
        },
        'disease_statistics': {
            'total_diseases': int(disease_counts.nunique()),
            'diseases_in_train': int(len(train_disease_counts)),
            'diseases_in_val': int(len(val_disease_counts)),
            'diseases_in_test': int(len(test_disease_counts)),
            'min_samples_per_disease': int(min_samples),
            'max_samples_per_disease': int(max_samples),
            'mean_samples_per_disease': float(disease_counts.mean()),
            'median_samples_per_disease': float(disease_counts.median())
        },
        'class_weights': {
            'num_classes': len(class_weights_dict),
            'min_weight': float(weights_array.min()),
            'max_weight': float(weights_array.max()),
            'weight_ratio': float(weights_array.max() / weights_array.min())
        },
        'stratification': {
            'all_diseases_in_all_splits': all_in_all_splits,
            'max_deviation_percent': float(max_deviation * 100)
        },
        'outputs': {
            'train_csv': str(train_path),
            'val_csv': str(val_path),
            'test_csv': str(test_path),
            'class_weights_json': str(weights_path),
            'visualizations': str(viz_path)
        }
    }

    report_path = OUTPUT_DIR / 'preprocessing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"   ✅ {report_path}")
    print()

    # -------------------------------------------------------------------------
    # 9. SUMMARY
    # -------------------------------------------------------------------------
    print("="*80)
    print("✅ PREPROCESSING COMPLETE - COMMON DISEASES (577)")
    print("="*80)
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(common_data):,}")
    print(f"   Total diseases: {disease_counts.nunique()}")
    print(f"   Train: {len(train_df):,} samples ({len(train_df)/len(common_data)*100:.2f}%)")
    print(f"   Val:   {len(val_df):,} samples ({len(val_df)/len(common_data)*100:.2f}%)")
    print(f"   Test:  {len(test_df):,} samples ({len(test_df)/len(common_data)*100:.2f}%)")

    print(f"\n⚖️  Class Weights:")
    print(f"   Weight range: {weights_array.min():.3f} to {weights_array.max():.3f}")
    print(f"   Weight ratio: {weights_array.max() / weights_array.min():.1f}:1")

    print(f"\n✅ Stratification:")
    if all_in_all_splits:
        print(f"   All 577 diseases present in all splits ✓")
    print(f"   Max deviation: {max_deviation*100:.3f}%")

    print(f"\n💾 Outputs saved to: {OUTPUT_DIR}/")
    print(f"   - train.csv, val.csv, test.csv")
    print(f"   - class_weights.json")
    print(f"   - preprocessing_report.json")
    print(f"   - preprocessing_visualizations.png")

    print(f"\n{'='*80}")
    print("🚀 Ready for ANN training on 577 diseases!")
    print("   (191 rare diseases will be added later)")
    print("="*80)

if __name__ == "__main__":
    main()
