# 🏭 Industrial Defect Detection - Streamlit App

A simple, standalone Streamlit application for industrial defect detection using ensemble machine learning models with majority voting.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Using Your Own Models](#-using-your-own-models)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🖼️ **Image Upload** | Support for JPG, JPEG, PNG, BMP formats |
| 🧠 **Deep Feature Extraction** | ResNet50, VGG16, DenseNet121 pre-trained models |
| 🤖 **CNN Baseline** | Custom CNN architecture for direct classification |
| 📊 **Shallow Classifiers** | SVM, XGBoost, Random Forest on extracted features |
| 🗳️ **Majority Voting** | Ensemble of 5 models for robust predictions |
| 📈 **Rich Visualizations** | Confidence gauge, vote distribution pie chart, model comparison table |
| 🧪 **Demo Mode** | Test the UI without trained models |
| 🎨 **Modern UI** | Clean, responsive design with custom CSS |

---

## 🎬 Demo

### Demo Mode (No Models Required)

The app includes a built-in demo mode that simulates predictions, perfect for:
- Testing the user interface
- Understanding the workflow
- Hackathon demonstrations

### Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  🏭 Industrial Defect Detection                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Upload Image]              [Analysis Results]             │
│  ┌───────────┐              ┌─────────────────────────┐    │
│  │           │              │ ⚠️ DEFECTIVE            │    │
│  │   IMAGE   │              │ Confidence: 80% (4/5)   │    │
│  │           │              └─────────────────────────┘    │
│  └───────────┘                                             │
│                             [Vote Details] [Models] [Conf] │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download

```bash
# If you have the file
cd your-project-directory
```

### Step 2: Install Dependencies

```bash
pip install streamlit torch torchvision scikit-learn xgboost pillow plotly pandas joblib
```

Or create a `requirements.txt`:

```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
xgboost>=2.0.0
Pillow>=10.0.0
plotly>=5.17.0
pandas>=2.0.0
joblib>=1.3.0
```

Then run:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import streamlit; import torch; print('✅ All dependencies installed!')"
```

---

## ⚡ Quick Start

### Run the App

```bash
streamlit run streamlit_prediction_app.py
```

### Access the App

Open your browser and go to:
```
http://localhost:8501
```

### Basic Usage

1. ✅ Check "Demo Mode" in the sidebar (enabled by default)
2. 📤 Upload an image (JPG, PNG, or BMP)
3. 🚀 Click "Analyze Image"
4. 📊 View results in the three tabs

---

## 📁 Project Structure

```
project/
├── streamlit_prediction_app.py    # Main application
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── models/                        # Your trained models (optional)
    ├── baseline_cnn.pth          # PyTorch CNN model
    ├── resnet50_svm.pkl          # SVM classifier
    ├── resnet50_xgboost.pkl      # XGBoost classifier
    ├── vgg16_randomforest.pkl    # Random Forest classifier
    ├── densenet121_svm.pkl       # SVM classifier
    ├── resnet50_scaler.pkl       # Feature scaler
    ├── vgg16_scaler.pkl          # Feature scaler
    └── densenet121_scaler.pkl    # Feature scaler
```

---

## 🔧 How It Works

### Architecture Overview

```
                            ┌─────────────────────────┐
                            │      INPUT IMAGE        │
                            │      (224 × 224)        │
                            └───────────┬─────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
            │   CNN         │   │   ResNet50    │   │   VGG16       │
            │   Baseline    │   │   Extractor   │   │   Extractor   │
            └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                    │                   │                   │
                    │           ┌───────┴───────┐           │
                    │           │               │           │
                    │           ▼               ▼           ▼
                    │       ┌───────┐       ┌───────┐   ┌───────┐
                    │       │  SVM  │       │XGBoost│   │  RF   │
                    │       └───┬───┘       └───┬───┘   └───┬───┘
                    │           │               │           │
                    └───────────┴───────┬───────┴───────────┘
                                        │
                                        ▼
                            ┌─────────────────────────┐
                            │    MAJORITY VOTING      │
                            │    (5 models vote)      │
                            └───────────┬─────────────┘
                                        │
                                        ▼
                            ┌─────────────────────────┐
                            │   FINAL PREDICTION      │
                            │   Defective / OK        │
                            │   + Confidence Score    │
                            └─────────────────────────┘
```

### Ensemble Models

| Model Name | Feature Extractor | Classifier | Output Dim |
|------------|-------------------|------------|------------|
| CNN_Baseline | - | Custom CNN | 2 |
| ResNet50_SVM | ResNet50 | SVM (RBF) | 2048 |
| ResNet50_XGBoost | ResNet50 | XGBoost | 2048 |
| VGG16_RandomForest | VGG16 | Random Forest | 25088 |
| DenseNet121_SVM | DenseNet121 | SVM (Linear) | 1024 |

### Majority Voting Logic

```python
# Example: 5 models vote
predictions = [1, 1, 0, 1, 1]  # 1 = Defective, 0 = OK

defective_votes = sum(predictions)  # 4
total_models = len(predictions)      # 5

# Majority wins
if defective_votes > total_models / 2:
    final_prediction = "Defective"
    confidence = defective_votes / total_models  # 80%
else:
    final_prediction = "Non-Defective"
    confidence = (total_models - defective_votes) / total_models
```

---

## 🎯 Using Your Own Models

### Step 1: Train Your Models

Use the training notebook or your own training pipeline to create:

1. **CNN Model** (PyTorch)
```python
# Save CNN
torch.save(model.state_dict(), 'models/baseline_cnn.pth')
```

2. **Shallow Classifiers** (scikit-learn/XGBoost)
```python
# Save classifier
joblib.dump(svm_model, 'models/resnet50_svm.pkl')
joblib.dump(xgb_model, 'models/resnet50_xgboost.pkl')
```

3. **Feature Scalers** (IMPORTANT!)
```python
# Save scaler - must match the extractor name
joblib.dump(scaler, 'models/resnet50_scaler.pkl')
```

### Step 2: Naming Convention

Models must follow this naming pattern:

```
{extractor}_{classifier}.pkl
```

Examples:
- `resnet50_svm.pkl` → Uses ResNet50 features + SVM
- `vgg16_randomforest.pkl` → Uses VGG16 features + Random Forest
- `densenet121_xgboost.pkl` → Uses DenseNet121 features + XGBoost

Scalers must match:
- `resnet50_scaler.pkl`
- `vgg16_scaler.pkl`
- `densenet121_scaler.pkl`

### Step 3: Disable Demo Mode

In the app sidebar:
- ❌ Uncheck "Demo Mode"
- The app will automatically load models from `./models/`

---

## ⚙️ Configuration

### Change Models Directory

Edit the `MODELS_DIR` constant in the script:

```python
MODELS_DIR = Path("./models")  # Change to your path
```

### Add New Feature Extractors

Add to the `FeatureExtractor` class:

```python
elif model_name == 'efficientnet_b0':
    base = models.efficientnet_b0(pretrained=True)
    self.features = base.features
    self.output_dim = 1280
```

### Customize UI Colors

Edit the CSS in the `st.markdown()` section:

```css
.result-defective {
    background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
    border-left: 6px solid #F44336;
}
```

---

## 📚 API Reference

### Main Functions

#### `ensemble_predict(image_tensor, extractors, models_dict)`

Makes ensemble prediction using all loaded models.

**Parameters:**
- `image_tensor`: PyTorch tensor (1, 3, 224, 224)
- `extractors`: Dict of feature extractors
- `models_dict`: Dict containing CNN models, shallow models, and scalers

**Returns:**
```python
{
    'prediction': 0 or 1,
    'class_name': 'Defective' or 'Non-Defective',
    'confidence': float (0.0 - 1.0),
    'num_models': int,
    'votes': {'defective': int, 'non_defective': int},
    'model_results': {
        'model_name': {
            'prediction': int,
            'confidence': float,
            'class_name': str
        }
    }
}
```

#### `predict_with_cnn(model, image_tensor)`

Makes prediction with a CNN model.

**Returns:** `(prediction: int, confidence: float)`

#### `predict_with_shallow(model, features, scaler=None)`

Makes prediction with a shallow classifier.

**Returns:** `(prediction: int, confidence: float)`

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "No module named 'torch'"
```bash
pip install torch torchvision
```

#### 2. "CUDA out of memory"
The app defaults to CPU if CUDA is not available. Check:
```python
print(torch.cuda.is_available())
```

#### 3. "Models not loading"
Ensure models are in the correct directory and follow naming conventions:
```bash
ls -la ./models/
```

#### 4. "Streamlit not found"
```bash
pip install streamlit --upgrade
```

#### 5. "Image upload fails"
Supported formats: JPG, JPEG, PNG, BMP
Maximum size: depends on your system memory

### Performance Tips

1. **Use GPU**: If available, models will automatically use CUDA
2. **Reduce models**: Comment out models you don't need in `create_demo_models()`
3. **Resize images**: Large images are automatically resized to 224×224

---

## 📄 License

MIT License - Feel free to use and modify for your hackathon!

---

## 🙏 Acknowledgments

- **PyTorch** for deep learning framework
- **Streamlit** for the amazing web app framework
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning utilities

---

## 📞 Support

For hackathon support:
- 📧 Check the technical guide
- 💬 Ask your facilitator
- 📖 Refer to the training notebook

---

**Built with ❤️ for the Industrial Defect Detection Hackathon 2026, Skema Business School**
