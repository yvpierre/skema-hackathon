"""
💾 Quick Reference: How to Save Models
======================================
Simple examples of how to save each type of model.
Copy-paste these snippets into your training notebook.
"""

import torch
import joblib
import json
from pathlib import Path

# ============================================
# SETUP
# ============================================

MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)


# ============================================
# 1. SAVE CNN MODEL (PyTorch)
# ============================================

def save_cnn_model(model, filename='baseline_cnn.pth'):
    """
    Save a PyTorch CNN model.
    
    Example:
        model = BaselineCNN()
        # ... train model ...
        save_cnn_model(model, 'baseline_cnn.pth')
    """
    path = MODELS_DIR / filename
    
    # Method 1: Save state_dict (RECOMMENDED)
    torch.save(model.state_dict(), path)
    
    # Method 2: Save entire model (less flexible)
    # torch.save(model, path)
    
    print(f"✅ Saved CNN: {path}")
    return path


def load_cnn_model(model_class, filename='baseline_cnn.pth'):
    """
    Load a PyTorch CNN model.
    
    Example:
        model = load_cnn_model(BaselineCNN, 'baseline_cnn.pth')
    """
    path = MODELS_DIR / filename
    
    model = model_class()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Loaded CNN: {path}")
    return model


# ============================================
# 2. SAVE SKLEARN CLASSIFIERS
# ============================================

def save_sklearn_model(model, filename):
    """
    Save sklearn model (SVM, RandomForest, etc.)
    
    Example:
        from sklearn.svm import SVC
        svm = SVC(kernel='rbf')
        svm.fit(X_train, y_train)
        save_sklearn_model(svm, 'resnet50_svm.pkl')
    """
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"✅ Saved classifier: {path}")
    return path


def load_sklearn_model(filename):
    """
    Load sklearn model.
    
    Example:
        svm = load_sklearn_model('resnet50_svm.pkl')
        predictions = svm.predict(X_test)
    """
    path = MODELS_DIR / filename
    model = joblib.load(path)
    print(f"✅ Loaded classifier: {path}")
    return model


# ============================================
# 3. SAVE XGBOOST MODELS
# ============================================

def save_xgboost_model(model, filename):
    """
    Save XGBoost model.
    
    Example:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train)
        save_xgboost_model(xgb_model, 'resnet50_xgboost.pkl')
    """
    path = MODELS_DIR / filename
    
    # Method 1: Using joblib (compatible with sklearn pipeline)
    joblib.dump(model, path)
    
    # Method 2: Using native XGBoost format
    # model.save_model(str(path).replace('.pkl', '.json'))
    
    print(f"✅ Saved XGBoost: {path}")
    return path


# ============================================
# 4. SAVE SCALERS (VERY IMPORTANT!)
# ============================================

def save_scaler(scaler, extractor_name):
    """
    Save feature scaler. 
    
    ⚠️ CRITICAL: You MUST save the scaler used during training!
    The same scaler must be used for inference.
    
    Example:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        save_scaler(scaler, 'resnet50')  # Saves as resnet50_scaler.pkl
    """
    filename = f'{extractor_name}_scaler.pkl'
    path = MODELS_DIR / filename
    joblib.dump(scaler, path)
    print(f"✅ Saved scaler: {path}")
    return path


def load_scaler(extractor_name):
    """
    Load feature scaler.
    
    Example:
        scaler = load_scaler('resnet50')
        X_test_scaled = scaler.transform(X_test)
    """
    filename = f'{extractor_name}_scaler.pkl'
    path = MODELS_DIR / filename
    scaler = joblib.load(path)
    print(f"✅ Loaded scaler: {path}")
    return scaler


# ============================================
# 5. SAVE MODEL REGISTRY (OPTIONAL)
# ============================================

def save_registry(models_info):
    """
    Save a registry of all trained models.
    
    Example:
        registry = {
            'CNN_Baseline': {'type': 'cnn', 'path': 'baseline_cnn.pth'},
            'resnet50_svm': {'type': 'shallow', 'extractor': 'resnet50', 'path': 'resnet50_svm.pkl'},
        }
        save_registry(registry)
    """
    path = MODELS_DIR / 'model_registry.json'
    with open(path, 'w') as f:
        json.dump(models_info, f, indent=2)
    print(f"✅ Saved registry: {path}")
    return path


# ============================================
# COMPLETE EXAMPLE
# ============================================

def complete_training_example():
    """
    Complete example of training and saving all models.
    """
    import numpy as np
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # ----------------------------------------
    # Simulate data (replace with your real data)
    # ----------------------------------------
    X_train = np.random.randn(100, 2048)  # 100 samples, 2048 features (ResNet50)
    y_train = np.random.randint(0, 2, 100)  # Binary labels
    
    X_val = np.random.randn(20, 2048)
    y_val = np.random.randint(0, 2, 20)
    
    # ----------------------------------------
    # Step 1: Scale features
    # ----------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # SAVE THE SCALER!
    save_scaler(scaler, 'resnet50')
    
    # ----------------------------------------
    # Step 2: Train SVM
    # ----------------------------------------
    svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Evaluate
    accuracy = svm.score(X_val_scaled, y_val)
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    # SAVE SVM
    save_sklearn_model(svm, 'resnet50_svm.pkl')
    
    # ----------------------------------------
    # Step 3: Train Random Forest
    # ----------------------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # SAVE RF
    save_sklearn_model(rf, 'resnet50_randomforest.pkl')
    
    # ----------------------------------------
    # Step 4: Save registry
    # ----------------------------------------
    registry = {
        'resnet50_svm': {
            'type': 'shallow',
            'classifier': 'svm',
            'extractor': 'resnet50',
            'path': 'resnet50_svm.pkl',
            'scaler': 'resnet50_scaler.pkl'
        },
        'resnet50_randomforest': {
            'type': 'shallow',
            'classifier': 'randomforest',
            'extractor': 'resnet50',
            'path': 'resnet50_randomforest.pkl',
            'scaler': 'resnet50_scaler.pkl'
        }
    }
    save_registry(registry)
    
    print("\n✅ All models saved successfully!")
    print(f"📁 Check: {MODELS_DIR}")


# ============================================
# NAMING CONVENTION REMINDER
# ============================================

"""
📝 NAMING CONVENTION FOR STREAMLIT APP
======================================

The Streamlit app expects models named as:

CNN Models:
    - baseline_cnn.pth          (PyTorch state_dict)

Shallow Classifiers:
    - {extractor}_{classifier}.pkl
    
    Examples:
    - resnet50_svm.pkl
    - resnet50_xgboost.pkl
    - vgg16_randomforest.pkl
    - densenet121_svm.pkl

Scalers (REQUIRED for shallow classifiers):
    - {extractor}_scaler.pkl
    
    Examples:
    - resnet50_scaler.pkl
    - vgg16_scaler.pkl
    - densenet121_scaler.pkl

Final directory structure:
    models/
    ├── baseline_cnn.pth
    ├── resnet50_svm.pkl
    ├── resnet50_xgboost.pkl
    ├── resnet50_scaler.pkl
    ├── vgg16_randomforest.pkl
    ├── vgg16_scaler.pkl
    ├── densenet121_svm.pkl
    ├── densenet121_scaler.pkl
    └── model_registry.json (optional)
"""


# ============================================
# RUN EXAMPLE
# ============================================

if __name__ == '__main__':
    print("=" * 50)
    print("💾 MODEL SAVING EXAMPLES")
    print("=" * 50)
    
    complete_training_example()
