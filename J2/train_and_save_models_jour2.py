"""
🏭 Model Training & Saving Script
==================================
Train and save all models for the Streamlit ensemble prediction app.

This script will:
1. Train a CNN Baseline model
2. Extract features with pre-trained CNNs (ResNet50, VGG16, DenseNet121)
3. Train shallow classifiers (SVM, XGBoost, RandomForest)
4. Save all models and scalers in the correct format

Usage:
    python train_and_save_models.py --data_dir ./data --output_dir ./models

Requirements:
    pip install torch torchvision scikit-learn xgboost pillow tqdm pandas
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Skipping XGBoost models.")


# ============================================
# CONFIGURATION
# ============================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASSES = ['non_defective', 'defective']


# ============================================
# DATASET
# ============================================

class DefectDataset(Dataset):
    """Dataset for defect images."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # Load all images
        for label, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((str(img_path), label))
        
        print(f"📁 Loaded {len(self.samples)} images from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_transforms():
    """Get train and test transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


# ============================================
# CNN BASELINE MODEL
# ============================================

class BaselineCNN(nn.Module):
    """Custom CNN for defect classification."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_cnn(model, train_loader, val_loader, epochs=20, lr=0.001, patience=5):
    """Train CNN model with early stopping."""
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history


# ============================================
# FEATURE EXTRACTORS
# ============================================

class FeatureExtractor(nn.Module):
    """Extract features from pre-trained models."""
    
    def __init__(self, model_name='resnet50'):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            base = models.resnet50(pretrained=True)
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.output_dim = 2048
            
        elif model_name == 'vgg16':
            base = models.vgg16(pretrained=True)
            self.features = base.features
            self.avgpool = base.avgpool
            self.output_dim = 512 * 7 * 7
            
        elif model_name == 'densenet121':
            base = models.densenet121(pretrained=True)
            self.features = base.features
            self.output_dim = 1024
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg16':
            x = self.avgpool(x)
        elif self.model_name == 'densenet121':
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


def extract_features(dataloader, extractor, device=DEVICE):
    """Extract features from all images in dataloader."""
    
    extractor = extractor.to(device)
    extractor.eval()
    
    all_features = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = extractor(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    
    return features, labels, all_paths


# ============================================
# TRAINING SHALLOW CLASSIFIERS
# ============================================

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM with grid search."""
    
    print("🔧 Training SVM...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"   Best params: {grid.best_params_}")
    print(f"   Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return best_model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    
    if not HAS_XGBOOST:
        return None
    
    print("🔧 Training XGBoost...")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"   Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return model


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier."""
    
    print("🔧 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"   Val Accuracy: {acc:.4f}, F1: {f1:.4f}")
    
    return model


# ============================================
# SAVE FUNCTIONS
# ============================================

def save_cnn_model(model, output_dir, name='baseline_cnn'):
    """Save PyTorch CNN model."""
    
    path = Path(output_dir) / f'{name}.pth'
    torch.save(model.state_dict(), path)
    print(f"✅ Saved CNN model: {path}")
    return str(path)


def save_shallow_model(model, output_dir, name):
    """Save sklearn/xgboost model."""
    
    path = Path(output_dir) / f'{name}.pkl'
    joblib.dump(model, path)
    print(f"✅ Saved classifier: {path}")
    return str(path)


def save_scaler(scaler, output_dir, extractor_name):
    """Save feature scaler."""
    
    path = Path(output_dir) / f'{extractor_name}_scaler.pkl'
    joblib.dump(scaler, path)
    print(f"✅ Saved scaler: {path}")
    return str(path)


def save_model_registry(registry, output_dir):
    """Save model registry JSON."""
    
    path = Path(output_dir) / 'model_registry.json'
    with open(path, 'w') as f:
        json.dump(registry, f, indent=2)
    print(f"✅ Saved registry: {path}")
    return str(path)


# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main(args):
    """Main training pipeline."""
    
    print("=" * 60)
    print("🏭 INDUSTRIAL DEFECT DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"💻 Device: {DEVICE}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get transforms
    train_transform, test_transform = get_transforms()
    
    # Load dataset
    print("\n📦 Loading dataset...")
    
    if args.train_dir and args.val_dir:
        # Separate train/val directories
        train_dataset = DefectDataset(args.train_dir, transform=train_transform)
        val_dataset = DefectDataset(args.val_dir, transform=test_transform)
    else:
        # Single directory, split automatically
        full_dataset = DefectDataset(args.data_dir, transform=test_transform)
        
        # Split indices
        indices = list(range(len(full_dataset)))
        labels = [full_dataset.samples[i][1] for i in indices]
        
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create subsets
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        
        # Apply different transform to train
        full_dataset_train = DefectDataset(args.data_dir, transform=train_transform)
        train_dataset = torch.utils.data.Subset(full_dataset_train, train_idx)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Model registry
    registry = {
        'created_at': datetime.now().isoformat(),
        'device': str(DEVICE),
        'models': {}
    }
    
    # ========================================
    # 1. TRAIN CNN BASELINE
    # ========================================
    
    if not args.skip_cnn:
        print("\n" + "=" * 60)
        print("🧠 TRAINING CNN BASELINE")
        print("=" * 60)
        
        cnn_model = BaselineCNN(num_classes=NUM_CLASSES)
        cnn_model, history = train_cnn(
            cnn_model, train_loader, val_loader,
            epochs=args.epochs, lr=args.lr, patience=args.patience
        )
        
        # Save CNN
        cnn_path = save_cnn_model(cnn_model, output_dir, 'baseline_cnn')
        registry['models']['CNN_Baseline'] = {
            'type': 'cnn',
            'path': cnn_path,
            'history': history
        }
    
    # ========================================
    # 2. EXTRACT FEATURES & TRAIN SHALLOW
    # ========================================
    
    # Feature extraction loader (no shuffle, no augmentation)
    full_dataset = DefectDataset(args.data_dir, transform=test_transform)
    feature_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    extractors_to_use = ['resnet50', 'vgg16', 'densenet121']
    
    for extractor_name in extractors_to_use:
        print("\n" + "=" * 60)
        print(f"🔬 FEATURE EXTRACTION WITH {extractor_name.upper()}")
        print("=" * 60)
        
        # Create extractor
        extractor = FeatureExtractor(extractor_name)
        
        # Extract features
        features, labels, paths = extract_features(feature_loader, extractor)
        print(f"   Features shape: {features.shape}")
        
        # Split features
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Save scaler
        scaler_path = save_scaler(scaler, output_dir, extractor_name)
        
        # Train SVM
        print(f"\n📊 Training classifiers for {extractor_name}...")
        
        svm_model = train_svm(X_train_scaled, y_train, X_val_scaled, y_val)
        svm_name = f'{extractor_name}_svm'
        svm_path = save_shallow_model(svm_model, output_dir, svm_name)
        registry['models'][svm_name] = {
            'type': 'shallow',
            'classifier': 'svm',
            'extractor': extractor_name,
            'path': svm_path,
            'scaler_path': scaler_path
        }
        
        # Train XGBoost (only for resnet50 to avoid too many models)
        if extractor_name == 'resnet50' and HAS_XGBOOST:
            xgb_model = train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
            if xgb_model:
                xgb_name = f'{extractor_name}_xgboost'
                xgb_path = save_shallow_model(xgb_model, output_dir, xgb_name)
                registry['models'][xgb_name] = {
                    'type': 'shallow',
                    'classifier': 'xgboost',
                    'extractor': extractor_name,
                    'path': xgb_path,
                    'scaler_path': scaler_path
                }
        
        # Train Random Forest (only for vgg16)
        if extractor_name == 'vgg16':
            rf_model = train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            rf_name = f'{extractor_name}_randomforest'
            rf_path = save_shallow_model(rf_model, output_dir, rf_name)
            registry['models'][rf_name] = {
                'type': 'shallow',
                'classifier': 'randomforest',
                'extractor': extractor_name,
                'path': rf_path,
                'scaler_path': scaler_path
            }
    
    # ========================================
    # 3. SAVE REGISTRY
    # ========================================
    
    save_model_registry(registry, output_dir)
    
    # ========================================
    # SUMMARY
    # ========================================
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Models saved to: {output_dir}")
    print(f"\n📊 Models trained: {len(registry['models'])}")
    
    for name, info in registry['models'].items():
        print(f"   - {name} ({info['type']})")
    
    print("\n📝 Files created:")
    for f in output_dir.glob('*'):
        print(f"   - {f.name}")
    
    print("\n🚀 You can now run the Streamlit app with:")
    print(f"   streamlit run streamlit_prediction_app.py")
    print(f"\n   Make sure MODELS_DIR points to: {output_dir}")


# ============================================
# CLI
# ============================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save defect detection models')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory (with non_defective/ and defective/ subfolders)')
    parser.add_argument('--train_dir', type=str, default=None,
                        help='Path to training data (optional, if separate from val)')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to validation data (optional)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Path to save trained models')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for CNN training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for CNN')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--skip_cnn', action='store_true',
                        help='Skip CNN training')
    
    args = parser.parse_args()
    main(args)
