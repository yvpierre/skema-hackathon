"""
🏭 Model Training & Saving Script
=================================
Supports data structure:
    data/
    ├── train/
    │   ├── defective/
    │   └── non_defective/
    └── test/
        ├── defective/
        └── non_defective/

Usage:
    # Simple (cherche dans ./data)
    python train_models_v2.py

    # Ou avec options
    python train_models_v2.py --data_dir ./data --output_dir ./models --epochs 20

"""

import os
import sys
import argparse
from pathlib import Path

# Check dependencies
def check_dependencies():
    missing = []
    try:
        import torch
    except ImportError:
        missing.append('torch torchvision')
    try:
        import sklearn
    except ImportError:
        missing.append('scikit-learn')
    try:
        from PIL import Image
    except ImportError:
        missing.append('Pillow')
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
    try:
        import tqdm
    except ImportError:
        missing.append('tqdm')
    
    if missing:
        print("❌ Missing dependencies! Run:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)
    print("✅ Dependencies OK")

check_dependencies()

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("✅ XGBoost available")
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed (optional)")

# ============================================
# CONFIG
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
BATCH_SIZE = 16
CLASSES = ['non_defective', 'defective']

print(f"💻 Device: {DEVICE}")

# ============================================
# CNN MODEL
# ============================================
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================
# FEATURE EXTRACTOR
# ============================================
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50'):
        super().__init__()
        self.model_name = model_name
        
        print(f"   Loading {model_name}...")
        
        if model_name == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.output_dim = 2048
        elif model_name == 'vgg16':
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.output_dim = 512 * 7 * 7
        elif model_name == 'densenet121':
            base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.output_dim = 1024
        
        for param in self.parameters():
            param.requires_grad = False
        
        print(f"   ✅ {model_name} loaded (dim: {self.output_dim})")
    
    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg16':
            x = self.avgpool(x)
        elif self.model_name == 'densenet121':
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

# ============================================
# DATASET
# ============================================
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def load_images_from_folder(folder_path):
    """Load images from folder with defective/non_defective subfolders."""
    folder = Path(folder_path)
    paths, labels = [], []
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = folder / class_name
        if not class_dir.exists():
            print(f"   ⚠️ Not found: {class_dir}")
            continue
        
        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            for img_path in class_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label_idx)
                count += 1
        
        print(f"   {class_name}: {count} images")
    
    return paths, labels


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

# ============================================
# TRAINING FUNCTIONS
# ============================================
def train_cnn(model, train_loader, val_loader, epochs=20, lr=0.001):
    print(f"\n🧠 Training CNN ({epochs} epochs)...")
    
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, pred = model(images).max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        
        acc = correct / total
        print(f"   Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"   ✅ Best Val Accuracy: {best_acc:.4f}")
    return model


def extract_features(dataloader, extractor):
    extractor = extractor.to(DEVICE)
    extractor.eval()
    
    all_features, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="   Extracting features", leave=False):
            features = extractor(images.to(DEVICE))
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.vstack(all_features), np.array(all_labels)


def train_and_save_classifiers(X_train, y_train, X_test, y_test, extractor_name, output_dir):
    """Train classifiers and save them."""
    
    # Scaler
    print(f"\n   Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = output_dir / f'{extractor_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Saved: {scaler_path.name}")
    
    # SVM
    print(f"\n   Training SVM...")
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    print(f"   SVM Results:")
    print(f"      Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"      F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    svm_path = output_dir / f'{extractor_name}_svm.pkl'
    joblib.dump(svm, svm_path)
    print(f"   ✅ Saved: {svm_path.name}")
    
    # XGBoost (only for resnet50)
    if HAS_XGBOOST and extractor_name == 'resnet50':
        print(f"\n   Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', random_state=42,
            verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        y_pred = xgb_model.predict(X_test_scaled)
        print(f"   XGBoost Results:")
        print(f"      Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"      F1-Score: {f1_score(y_test, y_pred):.4f}")
        
        xgb_path = output_dir / f'{extractor_name}_xgboost.pkl'
        joblib.dump(xgb_model, xgb_path)
        print(f"   ✅ Saved: {xgb_path.name}")
    
    # Random Forest (only for vgg16)
    if extractor_name == 'vgg16':
        print(f"\n   Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        
        y_pred = rf.predict(X_test_scaled)
        print(f"   RandomForest Results:")
        print(f"      Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"      F1-Score: {f1_score(y_test, y_pred):.4f}")
        
        rf_path = output_dir / f'{extractor_name}_randomforest.pkl'
        joblib.dump(rf, rf_path)
        print(f"   ✅ Saved: {rf_path.name}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--skip_cnn', action='store_true')
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🏭 DEFECT DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    
    print(f"\n📁 Data directory: {data_dir}")
    print(f"📁 Train folder: {train_dir}")
    print(f"📁 Test folder: {test_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Check structure
    if not train_dir.exists() or not test_dir.exists():
        print(f"\n❌ Expected structure not found!")
        print(f"   {data_dir}/")
        print(f"   ├── train/")
        print(f"   │   ├── defective/")
        print(f"   │   └── non_defective/")
        print(f"   └── test/")
        print(f"       ├── defective/")
        print(f"       └── non_defective/")
        return
    
    # Load data
    print("\n📂 Loading TRAIN data:")
    train_paths, train_labels = load_images_from_folder(train_dir)
    
    print("\n📂 Loading TEST data:")
    test_paths, test_labels = load_images_from_folder(test_dir)
    
    if len(train_paths) == 0 or len(test_paths) == 0:
        print("\n❌ No images found!")
        return
    
    print(f"\n📊 Total: {len(train_paths)} train, {len(test_paths)} test")
    
    # Transforms
    train_transform, test_transform = get_transforms()
    
    # Datasets
    train_dataset = ImageDataset(train_paths, train_labels, train_transform)
    test_dataset = ImageDataset(test_paths, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # For feature extraction (no augmentation)
    train_dataset_noaug = ImageDataset(train_paths, train_labels, test_transform)
    train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # ========================================
    # 1. CNN BASELINE
    # ========================================
    if not args.skip_cnn:
        print("\n" + "=" * 60)
        print("🧠 CNN BASELINE")
        print("=" * 60)
        
        cnn = BaselineCNN(num_classes=2)
        cnn = train_cnn(cnn, train_loader, test_loader, epochs=args.epochs, lr=args.lr)
        
        # Final evaluation
        cnn.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, pred = cnn(images).max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print(f"\n   📊 Final Test Results:")
        print(f"      Accuracy: {correct/total:.4f}")
        print(f"      F1-Score: {f1_score(all_labels, all_preds):.4f}")
        
        cnn_path = output_dir / 'baseline_cnn.pth'
        torch.save(cnn.state_dict(), cnn_path)
        print(f"   ✅ Saved: {cnn_path.name}")
    
    # ========================================
    # 2. FEATURE EXTRACTION + SHALLOW
    # ========================================
    for extractor_name in ['resnet50', 'vgg16', 'densenet121']:
        print("\n" + "=" * 60)
        print(f"🔬 {extractor_name.upper()}")
        print("=" * 60)
        
        extractor = FeatureExtractor(extractor_name)
        
        # Extract train features
        print("\n   Extracting TRAIN features...")
        X_train, y_train = extract_features(train_loader_noaug, extractor)
        print(f"   Train features: {X_train.shape}")
        
        # Extract test features
        print("\n   Extracting TEST features...")
        X_test, y_test = extract_features(test_loader, extractor)
        print(f"   Test features: {X_test.shape}")
        
        # Train classifiers
        train_and_save_classifiers(X_train, y_train, X_test, y_test, extractor_name, output_dir)
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    print(f"\n📁 Models saved to: {output_dir}/")
    for f in sorted(output_dir.glob('*')):
        size = f.stat().st_size / 1024 / 1024
        print(f"   - {f.name} ({size:.1f} MB)")
    
    print(f"\n🚀 Now run your Streamlit app:")
    print(f"   streamlit run streamlit_prediction_app.py")
    print(f"\n   Make sure MODELS_DIR = '{output_dir}' in the app")


if __name__ == '__main__':
    main()
