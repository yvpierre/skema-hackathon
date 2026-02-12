"""
Train VGG16 feature extractor + SVM classifier
"""
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import numpy as np

# Configuration
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = Path("./models")

def extract_features_vgg16(data_loader, device):
    """Extract features using VGG16"""
    # Load pretrained VGG16
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    vgg16 = vgg16.features
    vgg16.eval()
    vgg16.to(device)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = vgg16(images)
            avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
            features = avgpool(features)
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.vstack(all_features), np.array(all_labels)

def train_vgg16_svm():
    """Train SVM on VGG16 features"""
    print("🚀 Training VGG16 + SVM...")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load training data
    train_dataset = ImageFolder('./data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    
    print("📊 Extracting features from training data...")
    X_train, y_train = extract_features_vgg16(train_loader, DEVICE)
    
    print(f"✅ Extracted {X_train.shape[0]} training samples, {X_train.shape[1]} features")
    
    # Scale features
    print("📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train SVM
    print("🤖 Training SVM classifier...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate on training data
    train_acc = svm.score(X_train_scaled, y_train)
    print(f"✅ Training accuracy: {train_acc:.4f}")
    
    # Save models
    print("💾 Saving models...")
    joblib.dump(svm, MODELS_DIR / 'vgg16_svm.pkl')
    joblib.dump(scaler, MODELS_DIR / 'vgg16_scaler.pkl')
    
    print("✅ VGG16 + SVM model saved!")

if __name__ == "__main__":
    train_vgg16_svm()
