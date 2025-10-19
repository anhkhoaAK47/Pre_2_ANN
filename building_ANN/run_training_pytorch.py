#!/usr/bin/env python3
"""
PyTorch ANN Training Script for Common Diseases (577 diseases)
==============================================================

This script trains the ANN model using PyTorch for 577-disease classification.
Uses only common diseases (>10 samples each) for reliable evaluation.

The 191 rare diseases will be added later when more data is available.

Usage:
    python run_training_pytorch.py

Outputs:
    - models/pytorch/ann_disease_classifier_577.pth
    - models/pytorch/best_model.pth
    - models/pytorch/label_encoder.pkl
    - models/pytorch/training_history.csv
    - models/pytorch/training_report.json
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
import pickle
from tqdm import tqdm  # Using standard tqdm (not tqdm.notebook)

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path('../data/processed/common')
MODEL_DIR = Path('../models/pytorch')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CLASS
# ============================================================================

class DiseaseDataset(Dataset):
    """PyTorch Dataset for disease classification."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ANNDiseaseClassifier(nn.Module):
    """
    ANN model for 577-disease classification.

    Architecture:
    - Input: 377 features
    - Hidden Layer 1: 512 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Hidden Layer 2: 256 neurons + BatchNorm + ReLU + Dropout(0.3)
    - Hidden Layer 3: 128 neurons + BatchNorm + ReLU + Dropout(0.2)
    - Output: 577 classes + Softmax
    """

    def __init__(self, input_dim, num_classes, dropout_rates=(0.3, 0.3, 0.2)):
        super(ANNDiseaseClassifier, self).__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rates[0])

        # Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rates[1])

        # Layer 3
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rates[2])

        # Output layer
        self.fc4 = nn.Linear(128, num_classes)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)
        return x  # CrossEntropyLoss includes softmax

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(train_loader, desc='Training', leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Predictions
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc='Validation', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # Predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    # Calculate top-k accuracies
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3, labels=np.arange(all_probs.shape[1]))
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5, labels=np.arange(all_probs.shape[1]))

    return epoch_loss, epoch_acc, top3_acc, top5_acc

def plot_training_history(history, save_path):
    """Create training history visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Top-3 Accuracy
    axes[1, 0].plot(epochs, history['val_top3_acc'], label='Validation Top-3', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Top-3 Accuracy', fontsize=12)
    axes[1, 0].set_title('Top-3 Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Top-5 Accuracy
    axes[1, 1].plot(epochs, history['val_top5_acc'], label='Validation Top-5', linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Top-5 Accuracy', fontsize=12)
    axes[1, 1].set_title('Top-5 Accuracy During Training', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("PYTORCH ANN TRAINING - 577 COMMON DISEASE CLASSIFICATION")
    print("="*80)
    print(f"Strategy: Train on diseases with >10 samples")
    print(f"Rare diseases (191) will be added later")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print()

    # -------------------------------------------------------------------------
    # 1. LOAD PREPROCESSED DATA
    # -------------------------------------------------------------------------
    print("📂 Step 1: Loading preprocessed data...")
    train_df = pd.read_csv(DATA_DIR / 'train.csv')
    val_df = pd.read_csv(DATA_DIR / 'val.csv')
    test_df = pd.read_csv(DATA_DIR / 'test.csv')

    print(f"   Training set:   {len(train_df):,} samples")
    print(f"   Validation set: {len(val_df):,} samples")
    print(f"   Test set:       {len(test_df):,} samples")
    print()

    # -------------------------------------------------------------------------
    # 2. PREPARE FEATURES AND LABELS
    # -------------------------------------------------------------------------
    print("🔧 Step 2: Preparing features and labels...")

    # Separate features and labels
    X_train = train_df.drop('disease', axis=1).values.astype('float32')
    y_train_labels = train_df['disease'].values

    X_val = val_df.drop('disease', axis=1).values.astype('float32')
    y_val_labels = val_df['disease'].values

    X_test = test_df.drop('disease', axis=1).values.astype('float32')
    y_test_labels = test_df['disease'].values

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['disease'])

    y_train = label_encoder.transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    y_test = label_encoder.transform(y_test_labels)

    input_dim = X_train.shape[1]
    num_classes = len(label_encoder.classes_)

    print(f"   Input features: {input_dim}")
    print(f"   Output classes: {num_classes}")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print()

    # Save label encoder
    with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"   ✅ Label encoder saved")
    print()

    # -------------------------------------------------------------------------
    # 3. LOAD CLASS WEIGHTS
    # -------------------------------------------------------------------------
    print("⚖️  Step 3: Loading class weights...")
    with open(DATA_DIR / 'class_weights.json', 'r', encoding='utf-8') as f:
        class_weights_dict = json.load(f)

    # Convert to tensor
    class_weights = torch.zeros(num_classes)
    for disease_name, weight in class_weights_dict.items():
        class_idx = label_encoder.transform([disease_name])[0]
        class_weights[class_idx] = weight

    class_weights = class_weights.to(DEVICE)

    print(f"   Class weights loaded: {num_classes} classes")
    print(f"   Weight range: {class_weights.min().item():.3f} to {class_weights.max().item():.3f}")
    print(f"   Weight ratio: {class_weights.max().item() / class_weights.min().item():.1f}:1")
    print()

    # -------------------------------------------------------------------------
    # 4. CREATE DATALOADERS
    # -------------------------------------------------------------------------
    print("📦 Step 4: Creating DataLoaders...")

    train_dataset = DiseaseDataset(X_train, y_train)
    val_dataset = DiseaseDataset(X_val, y_val)
    test_dataset = DiseaseDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print()

    # -------------------------------------------------------------------------
    # 5. BUILD MODEL
    # -------------------------------------------------------------------------
    print("🏗️  Step 5: Building PyTorch ANN model...")
    model = ANNDiseaseClassifier(input_dim, num_classes)
    model = model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Architecture: {input_dim} → 512 → 256 → 128 → {num_classes}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print()

    # -------------------------------------------------------------------------
    # 6. SETUP TRAINING
    # -------------------------------------------------------------------------
    print("⚙️  Step 6: Setting up training...")

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler (verbose removed for newer PyTorch versions)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    print(f"   Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   Loss: CrossEntropyLoss (weighted)")
    print(f"   Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    print()

    # -------------------------------------------------------------------------
    # 7. TRAIN MODEL
    # -------------------------------------------------------------------------
    print("="*80)
    print("🚀 Step 7: Training model...")
    print("="*80)
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max epochs: {EPOCHS}")
    print(f"Device: {DEVICE}")
    print()

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top3_acc': [],
        'val_top5_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 15

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc, val_top3_acc, val_top5_acc = validate_epoch(model, val_loader, criterion, DEVICE)

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3_acc'].append(val_top3_acc)
        history['val_top5_acc'].append(val_top5_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        print(f"Val Top-3: {val_top3_acc*100:.2f}% | Val Top-5: {val_top5_acc*100:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, MODEL_DIR / 'best_model.pth')
            print(f"✅ Best model saved (Val Acc: {val_acc*100:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience_limit:
            print(f"\n⚠️  Early stopping triggered (patience={patience_limit})")
            break

    print()
    print("="*80)
    print("✅ Training completed!")
    print("="*80)
    print()

    # -------------------------------------------------------------------------
    # 8. SAVE FINAL MODEL
    # -------------------------------------------------------------------------
    print("💾 Step 8: Saving final model...")

    final_model_path = MODEL_DIR / 'ann_disease_classifier_577.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'num_classes': num_classes,
        'architecture': '377 -> 512 -> 256 -> 128 -> 577'
    }, final_model_path)
    print(f"   ✅ Final model: {final_model_path}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(MODEL_DIR / 'training_history.csv', index=False)
    print(f"   ✅ Training history: {MODEL_DIR / 'training_history.csv'}")
    print()

    # -------------------------------------------------------------------------
    # 9. VISUALIZE TRAINING
    # -------------------------------------------------------------------------
    print("📊 Step 9: Creating visualizations...")
    plot_path = MODEL_DIR / 'training_history_curves.png'
    plot_training_history(history, plot_path)
    print(f"   ✅ Training curves: {plot_path}")
    print()

    # -------------------------------------------------------------------------
    # 10. EVALUATE ON TEST SET
    # -------------------------------------------------------------------------
    print("🧪 Step 10: Evaluating on test set...")

    # Load best model
    checkpoint = torch.load(MODEL_DIR / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, test_top3_acc, test_top5_acc = validate_epoch(model, test_loader, criterion, DEVICE)

    print(f"\n{'='*80}")
    print(f"📊 TEST SET PERFORMANCE")
    print(f"{'='*80}")
    print(f"Test Loss:         {test_loss:.4f}")
    print(f"Test Accuracy:     {test_acc*100:.2f}%")
    print(f"Test Top-3 Acc:    {test_top3_acc*100:.2f}%")
    print(f"Test Top-5 Acc:    {test_top5_acc*100:.2f}%")
    print(f"{'='*80}\n")

    # -------------------------------------------------------------------------
    # 11. GENERATE REPORT
    # -------------------------------------------------------------------------
    print("📋 Step 11: Generating training report...")

    best_epoch = np.argmax(history['val_acc'])

    training_report = {
        'model_info': {
            'framework': 'PyTorch',
            'architecture': 'ANN (377 -> 512 -> 256 -> 128 -> 577)',
            'input_features': int(input_dim),
            'output_classes': int(num_classes),
            'total_parameters': int(total_params),
            'note': 'Trained on 577 common diseases (>10 samples each). 191 rare diseases excluded.'
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'epochs_run': len(history['train_loss']),
            'optimizer': f'Adam (lr={LEARNING_RATE})',
            'loss_function': 'CrossEntropyLoss (weighted)',
            'class_weighted': True,
            'random_seed': RANDOM_SEED,
            'device': str(DEVICE)
        },
        'dataset_info': {
            'training_samples': int(len(X_train)),
            'validation_samples': int(len(X_val)),
            'test_samples': int(len(X_test)),
            'total_samples': int(len(X_train) + len(X_val) + len(X_test)),
            'rare_diseases_excluded': 191
        },
        'best_epoch_metrics': {
            'best_epoch': int(best_epoch + 1),
            'train_loss': float(history['train_loss'][best_epoch]),
            'train_accuracy': float(history['train_acc'][best_epoch]),
            'val_loss': float(history['val_loss'][best_epoch]),
            'val_accuracy': float(history['val_acc'][best_epoch])
        },
        'final_test_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_top_3_accuracy': float(test_top3_acc),
            'test_top_5_accuracy': float(test_top5_acc)
        },
        'outputs': {
            'model_file': str(final_model_path),
            'best_model_file': str(MODEL_DIR / 'best_model.pth'),
            'label_encoder': str(MODEL_DIR / 'label_encoder.pkl'),
            'training_history': str(MODEL_DIR / 'training_history.csv'),
            'visualizations': str(plot_path)
        }
    }

    # Save report
    report_path = MODEL_DIR / 'training_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Training report: {report_path}")
    print()

    # -------------------------------------------------------------------------
    # 12. SUMMARY
    # -------------------------------------------------------------------------
    print("="*80)
    print("🎉 TRAINING SUMMARY")
    print("="*80)
    print(f"\n🏗️  Model Architecture:")
    print(f"   PyTorch ANN: Input → 512 → 256 → 128 → Output (577 classes)")
    print(f"   Total parameters: {total_params:,}")
    print(f"\n📊 Training:")
    print(f"   Epochs run: {len(history['train_loss'])}")
    print(f"   Best epoch: {best_epoch + 1}")
    print(f"\n🎯 Best Performance (Validation):")
    print(f"   Accuracy: {history['val_acc'][best_epoch]*100:.2f}%")
    print(f"   Loss: {history['val_loss'][best_epoch]:.4f}")
    print(f"\n🧪 Final Test Performance:")
    print(f"   Accuracy:     {test_acc*100:.2f}%")
    print(f"   Top-3 Acc:    {test_top3_acc*100:.2f}%")
    print(f"   Top-5 Acc:    {test_top5_acc*100:.2f}%")
    print(f"\n💾 Outputs saved to: {MODEL_DIR}/")
    print(f"   - ann_disease_classifier_577.pth")
    print(f"   - best_model.pth")
    print(f"   - label_encoder.pkl")
    print(f"   - training_history.csv")
    print(f"   - training_report.json")
    print(f"   - training_history_curves.png")
    print(f"\n{'='*80}")
    print("🚀 Ready for model evaluation!")
    print(f"   Note: Model trained on 577 diseases")
    print(f"   191 rare diseases will be added later")
    print("="*80)

if __name__ == "__main__":
    main()
