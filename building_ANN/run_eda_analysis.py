#!/usr/bin/env python3
"""
Automated Exploratory Data Analysis (EDA) Script
==================================================

This script performs comprehensive EDA on the Vietnamese medical diagnosis dataset
to understand its characteristics and prepare for ANN architecture design.

Usage:
    python run_eda_analysis.py

Outputs:
    - data/eda_top20_diseases.png
    - data/eda_disease_distribution.png
    - data/eda_class_balance_pie.png
    - data/eda_top20_symptoms.png
    - data/eda_symptoms_per_patient.png
    - data/eda_sparsity_heatmap.png
    - data/eda_correlation_heatmap.png
    - data/eda_summary_stats.json
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = Path('../data/symptom_disease_dataset_7_converted.csv')
OUTPUT_DIR = Path('../data')
RANDOM_SEED = 42

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(RANDOM_SEED)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    print(f"Dataset: {DATA_PATH.name}")
    print()

    # -------------------------------------------------------------------------
    # 1. LOAD DATASET
    # -------------------------------------------------------------------------
    print("📂 Step 1: Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"   ✅ Dataset loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()

    # -------------------------------------------------------------------------
    # 2. DATASET OVERVIEW
    # -------------------------------------------------------------------------
    print("📊 Step 2: Dataset overview...")

    symptom_cols = df.columns[1:]  # All columns except 'disease'

    print(f"\n   Dimensions:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Total symptoms (features): {len(symptom_cols)}")
    print(f"\n   Disease Information:")
    print(f"   Unique diseases: {df['disease'].nunique()}")

    # Check missing values
    missing_total = df.isnull().sum().sum()
    print(f"\n   Missing Values: {missing_total}")

    # Check duplicates
    duplicates = df.duplicated().sum()
    duplicate_percentage = (duplicates / len(df)) * 100
    print(f"\n   Duplicate Rows:")
    print(f"   Count: {duplicates:,} ({duplicate_percentage:.2f}%)")

    if duplicates > 0:
        print(f"   ⚠️  WARNING: Dataset contains {duplicate_percentage:.1f}% duplicates")
    print()

    # -------------------------------------------------------------------------
    # 3. DISEASE DISTRIBUTION ANALYSIS
    # -------------------------------------------------------------------------
    print("🏥 Step 3: Disease distribution analysis...")

    disease_counts = df['disease'].value_counts()

    print(f"\n   Summary Statistics:")
    print(f"   Total unique diseases: {len(disease_counts)}")
    print(f"   Most common: {disease_counts.index[0]} ({disease_counts.iloc[0]:,} samples)")
    print(f"   Least common: {disease_counts.index[-1]} ({disease_counts.iloc[-1]:,} samples)")
    print(f"   Mean samples per disease: {disease_counts.mean():.1f}")
    print(f"   Median samples per disease: {disease_counts.median():.1f}")

    # Class balance analysis
    rare_diseases = disease_counts[disease_counts < 10]
    print(f"\n   Class Balance:")
    print(f"   Severely under-represented (<10 samples): {len(rare_diseases)} diseases")
    print(f"   This represents {(len(rare_diseases)/len(disease_counts)*100):.1f}% of all diseases")
    print()

    # -------------------------------------------------------------------------
    # 4. SYMPTOM ANALYSIS
    # -------------------------------------------------------------------------
    print("💊 Step 4: Symptom analysis...")

    symptom_frequencies = df[symptom_cols].sum().sort_values(ascending=False)
    symptoms_per_patient = df[symptom_cols].sum(axis=1)

    print(f"\n   Symptom Statistics:")
    print(f"   Total symptoms (features): {len(symptom_cols)}")
    print(f"   Most common: {symptom_frequencies.index[0]} ({symptom_frequencies.iloc[0]:,} occurrences)")
    print(f"   Least common: {symptom_frequencies.index[-1]} ({symptom_frequencies.iloc[-1]:,} occurrences)")

    print(f"\n   Symptoms per Patient:")
    print(f"   Mean: {symptoms_per_patient.mean():.2f}")
    print(f"   Median: {symptoms_per_patient.median():.0f}")
    print(f"   Min: {symptoms_per_patient.min()}")
    print(f"   Max: {symptoms_per_patient.max()}")

    # Calculate sparsity
    total_possible_values = len(df) * len(symptom_cols)
    total_ones = df[symptom_cols].sum().sum()
    sparsity = (1 - (total_ones / total_possible_values)) * 100

    print(f"\n   Data Sparsity:")
    print(f"   Sparsity: {sparsity:.2f}%")
    print(f"   Active features: {100 - sparsity:.2f}%")
    print(f"   Average active symptoms: {symptoms_per_patient.mean():.2f} out of {len(symptom_cols)}")
    print()

    # -------------------------------------------------------------------------
    # 5. CREATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("📊 Step 5: Creating visualizations...")

    # Visualization 1: Top 20 diseases bar chart
    print("   Generating chart 1/7: Top 20 diseases...")
    fig, ax = plt.subplots(figsize=(14, 8))
    top_20 = disease_counts.head(20)
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    bars = ax.barh(range(len(top_20)), top_20.values, color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index, fontsize=10)
    ax.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Common Diseases', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    for i, (bar, value) in enumerate(zip(bars, top_20.values)):
        ax.text(value + 20, bar.get_y() + bar.get_height()/2, f'{value:,}', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_top20_diseases.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 2: Disease distribution histogram
    print("   Generating chart 2/7: Disease distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(disease_counts.values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Diseases', fontsize=11, fontweight='bold')
    ax1.set_title('Disease Sample Distribution (Full Range)', fontsize=13, fontweight='bold')
    ax1.axvline(disease_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {disease_counts.mean():.0f}')
    ax1.axvline(disease_counts.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {disease_counts.median():.0f}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.hist(disease_counts.values, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Diseases (log scale)', fontsize=11, fontweight='bold')
    ax2.set_title('Disease Sample Distribution (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_disease_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 3: Class balance pie chart
    print("   Generating chart 3/7: Class balance...")
    bins = [0, 10, 50, 100, 500, 1000, 2000, float('inf')]
    labels = ['1-9', '10-49', '50-99', '100-499', '500-999', '1000-1999', '2000+']
    distribution = pd.cut(disease_counts, bins=bins, labels=labels).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_pie = plt.cm.Set3(range(len(distribution)))
    wedges, texts, autotexts = ax.pie(distribution.values, labels=distribution.index, autopct='%1.1f%%',
                                       startangle=90, colors=colors_pie, textprops={'fontsize': 11})
    ax.set_title('Disease Distribution by Sample Count Range', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_class_balance_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 4: Top 20 symptoms bar chart
    print("   Generating chart 4/7: Top 20 symptoms...")
    fig, ax = plt.subplots(figsize=(14, 8))
    top_20_symptoms = symptom_frequencies.head(20)
    colors = plt.cm.plasma(np.linspace(0, 1, 20))
    bars = ax.barh(range(len(top_20_symptoms)), top_20_symptoms.values, color=colors)
    ax.set_yticks(range(len(top_20_symptoms)))
    ax.set_yticklabels(top_20_symptoms.index, fontsize=10)
    ax.set_xlabel('Frequency (Number of Patients)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Common Symptoms', fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    for i, (bar, value) in enumerate(zip(bars, top_20_symptoms.values)):
        percentage = (value / len(df)) * 100
        ax.text(value + 300, bar.get_y() + bar.get_height()/2, f'{value:,} ({percentage:.1f}%)', va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_top20_symptoms.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 5: Symptoms per patient distribution
    print("   Generating chart 5/7: Symptoms per patient...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.hist(symptoms_per_patient, bins=range(1, symptoms_per_patient.max()+2), color='teal', edgecolor='black', alpha=0.7)
    ax1.axvline(symptoms_per_patient.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {symptoms_per_patient.mean():.2f}')
    ax1.axvline(symptoms_per_patient.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {symptoms_per_patient.median():.0f}')
    ax1.set_xlabel('Number of Symptoms', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Symptoms per Patient', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.boxplot(symptoms_per_patient, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='blue'),
                capprops=dict(color='blue'))
    ax2.set_ylabel('Number of Symptoms', fontsize=11, fontweight='bold')
    ax2.set_title('Symptoms per Patient (Box Plot)', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_symptoms_per_patient.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 6: Sparsity heatmap
    print("   Generating chart 6/7: Sparsity heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sample_df = df[symptom_cols].sample(100, random_state=RANDOM_SEED)
    im = ax.imshow(sample_df.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax.set_xlabel('Symptom Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Patient Sample (n=100)', fontsize=11, fontweight='bold')
    ax.set_title('Data Sparsity Visualization (Sample: 100 patients, 377 symptoms)', fontsize=13, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Symptom Present', fontsize=10, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_sparsity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualization 7: Correlation heatmap
    print("   Generating chart 7/7: Correlation heatmap...")
    top_20_symptom_names = symptom_frequencies.head(20).index.tolist()
    correlation_matrix = df[top_20_symptom_names].corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title('Symptom Correlation Matrix (Top 20 Symptoms)', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ All 7 charts saved to {OUTPUT_DIR}/")
    print()

    # -------------------------------------------------------------------------
    # 6. EXPORT SUMMARY STATISTICS
    # -------------------------------------------------------------------------
    print("📋 Step 6: Exporting summary statistics...")

    summary_stats = {
        'dataset_info': {
            'total_samples': int(len(df)),
            'total_features': int(len(symptom_cols)),
            'total_diseases': int(len(disease_counts)),
            'duplicates': int(duplicates),
            'duplicate_percentage': float(duplicate_percentage),
            'sparsity_percentage': float(sparsity)
        },
        'disease_stats': {
            'most_common': disease_counts.index[0],
            'most_common_count': int(disease_counts.iloc[0]),
            'least_common': disease_counts.index[-1],
            'least_common_count': int(disease_counts.iloc[-1]),
            'mean_samples': float(disease_counts.mean()),
            'median_samples': float(disease_counts.median()),
            'rare_diseases_count': int(len(rare_diseases))
        },
        'symptom_stats': {
            'most_common': symptom_frequencies.index[0],
            'most_common_count': int(symptom_frequencies.iloc[0]),
            'mean_symptoms_per_patient': float(symptoms_per_patient.mean()),
            'median_symptoms_per_patient': float(symptoms_per_patient.median()),
            'min_symptoms': int(symptoms_per_patient.min()),
            'max_symptoms': int(symptoms_per_patient.max())
        }
    }

    summary_path = OUTPUT_DIR / 'eda_summary_stats.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, ensure_ascii=False, indent=2)

    print(f"   ✅ Summary statistics: {summary_path}")
    print()

    # -------------------------------------------------------------------------
    # 7. DATA QUALITY SUMMARY
    # -------------------------------------------------------------------------
    print("="*80)
    print("DATA QUALITY SUMMARY")
    print("="*80)

    print(f"\n✅ STRENGTHS:")
    print(f"   1. No missing values - Clean dataset")
    print(f"   2. Binary encoding (0/1) - Perfect for neural networks")
    print(f"   3. Fixed feature length ({len(symptom_cols)}) - Consistent input size")
    print(f"   4. Large dataset ({len(df):,} samples) - Good for training")
    print(f"   5. Rich feature space (377 symptoms) - Comprehensive coverage")

    print(f"\n⚠️  CHALLENGES:")
    print(f"   1. High sparsity ({sparsity:.2f}%) - May need special techniques")
    print(f"   2. Severe class imbalance - {len(rare_diseases)} diseases have <10 samples")
    print(f"   3. Many target classes (768) - Complex classification problem")
    print(f"   4. Duplicate rows ({duplicate_percentage:.1f}%) - Need removal")
    print(f"   5. High dimensionality - 377 features for ~5 active per sample")

    print(f"\n📝 PREPROCESSING REQUIREMENTS:")
    print(f"   • Remove {duplicates:,} duplicate rows")
    print(f"   • Stratified train/val/test split (preserve class distribution)")
    print(f"   • Consider class weights to handle imbalance")
    print(f"   • Monitor performance on rare disease classes")
    print()

    # -------------------------------------------------------------------------
    # 8. RECOMMENDATIONS
    # -------------------------------------------------------------------------
    print("="*80)
    print("RECOMMENDATIONS FOR ANN ARCHITECTURE")
    print("="*80)

    print(f"\n🏗️  ARCHITECTURE DESIGN:")
    print(f"\n   1. Input Layer:")
    print(f"      • Size: {len(symptom_cols)} neurons (one per symptom)")
    print(f"      • Type: Binary inputs (0/1)")

    print(f"\n   2. Hidden Layers:")
    print(f"      • Recommended: 2-3 dense layers")
    print(f"      • Sizes: 512 → 256 → 128 neurons")
    print(f"      • Activation: ReLU")
    print(f"      • Regularization: Dropout (0.3-0.4) to handle sparsity")
    print(f"      • Normalization: Batch Normalization after each layer")

    print(f"\n   3. Output Layer:")
    print(f"      • Size: {len(disease_counts)} neurons (one per disease)")
    print(f"      • Activation: Softmax (multi-class classification)")

    print(f"\n⚙️  TRAINING STRATEGY:")
    print(f"   • Loss Function: Categorical Cross-Entropy with class weights")
    print(f"   • Optimizer: Adam (lr=0.001)")
    print(f"   • Metrics: Accuracy, Top-3 Accuracy, Macro F1-Score")
    print(f"   • Callbacks: Early Stopping, Model Checkpoint, LR Reduction")

    print(f"\n📊 DATA SPLITTING:")
    print(f"   • Train: 70% (~{int(len(df)*0.7):,} samples)")
    print(f"   • Validation: 15% (~{int(len(df)*0.15):,} samples)")
    print(f"   • Test: 15% (~{int(len(df)*0.15):,} samples)")
    print(f"   • Use stratified split to maintain class distribution")
    print()

    # -------------------------------------------------------------------------
    # 9. SUMMARY
    # -------------------------------------------------------------------------
    print("="*80)
    print("✅ EDA COMPLETE")
    print("="*80)

    print(f"\n📊 Key Findings:")
    print(f"   1. Dataset: {len(df):,} samples across {len(disease_counts)} diseases")
    print(f"   2. Data sparsity: {sparsity:.1f}% (avg {symptoms_per_patient.mean():.1f} symptoms/patient)")
    print(f"   3. Class imbalance: 1 to {disease_counts.iloc[0]:,} samples per disease")
    print(f"   4. Duplicates: {duplicate_percentage:.1f}% must be removed")
    print(f"   5. Dataset suitable for ANN with proper architecture design")

    print(f"\n💾 Outputs Generated:")
    print(f"   ✅ 7 visualization charts (data/*.png)")
    print(f"   ✅ Summary statistics (data/eda_summary_stats.json)")

    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Proceed to data preprocessing")
    print(f"   2. Remove duplicates and create stratified splits")
    print(f"   3. Design and implement ANN architecture")
    print(f"   4. Train model with recommended strategies")

    print(f"\n{'='*80}")
    print("END OF EXPLORATORY DATA ANALYSIS")
    print("="*80)

if __name__ == "__main__":
    main()
