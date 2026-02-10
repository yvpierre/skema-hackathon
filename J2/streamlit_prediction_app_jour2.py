"""
🏭 Industrial Defect Detection - Streamlit App
===============================================
Simple app for image upload, feature extraction, and ensemble prediction.
No CBIR, No VLM - Just pure classification with majority voting.

Usage:
    streamlit run streamlit_prediction_app.py

Requirements:
    pip install streamlit torch torchvision scikit-learn pillow plotly pandas
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from io import BytesIO
import time

# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="🏭 Détection de Défauts",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224

# Paths - Adjust these to your model locations
MODELS_DIR = Path("./models")

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .result-defective {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #F44336;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-defective h2, .result-defective p {
        color: #C62828 !important;
        margin: 0;
    }
    
    .result-ok {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-ok h2, .result-ok p {
        color: #2E7D32 !important;
        margin: 0;
    }
    
    .model-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .vote-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .vote-defect {
        background-color: #FFCDD2;
        color: #C62828;
    }
    
    .vote-ok {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    
    .stProgress > div > div > div > div {
        background-color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DEFINITIONS
# ============================================

class BaselineCNN(nn.Module):
    """Custom CNN for defect classification."""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
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


class FeatureExtractor(nn.Module):
    """Extract features from pre-trained CNN models."""
    
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
            
        elif model_name == 'mobilenet_v2':
            base = models.mobilenet_v2(pretrained=True)
            self.features = base.features
            self.output_dim = 1280
        
        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg16':
            x = self.avgpool(x)
        elif self.model_name == 'densenet121':
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        elif self.model_name == 'mobilenet_v2':
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


@st.cache_resource
def load_feature_extractors():
    """Load pre-trained feature extractors."""
    extractors = {}
    
    with st.spinner("Loading feature extractors..."):
        for name in ['resnet50', 'vgg16', 'densenet121']:
            try:
                extractor = FeatureExtractor(name)
                extractor.eval()
                extractor.to(DEVICE)
                extractors[name] = extractor
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    
    return extractors


@st.cache_resource
def load_models():
    """
    Load trained models.
    Returns dict of models and their associated extractors/scalers.
    """
    models_dict = {
        'cnn_models': {},
        'shallow_models': {}
    }
    
    # Try to load CNN baseline
    cnn_path = MODELS_DIR / 'baseline_cnn.pth'
    if cnn_path.exists():
        try:
            cnn = BaselineCNN()
            cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn.eval()
            cnn.to(DEVICE)
            models_dict['cnn_models']['CNN_Baseline'] = cnn
        except Exception as e:
            st.warning(f"Could not load CNN: {e}")
    
    # Try to load shallow classifiers
    for model_file in MODELS_DIR.glob('*.pkl'):
        if '_scaler' not in model_file.stem:
            try:
                model = joblib.load(model_file)
                models_dict['shallow_models'][model_file.stem] = {
                    'model': model,
                    'extractor': model_file.stem.split('_')[0]  # e.g., resnet50_svm -> resnet50
                }
            except Exception as e:
                pass
    
    # Load scalers
    models_dict['scalers'] = {}
    for scaler_file in MODELS_DIR.glob('*_scaler.pkl'):
        try:
            name = scaler_file.stem.replace('_scaler', '')
            models_dict['scalers'][name] = joblib.load(scaler_file)
        except:
            pass
    
    return models_dict


def create_demo_models():
    """
    Create demo models for testing when no trained models exist.
    These are NOT trained - just for UI demonstration!
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    
    # Create dummy classifiers (untrained - will predict randomly)
    demo_models = {
        'cnn_models': {
            'CNN_Baseline': BaselineCNN().eval().to(DEVICE)
        },
        'shallow_models': {
            'resnet50_SVM': {
                'model': SVC(kernel='rbf', probability=True),
                'extractor': 'resnet50'
            },
            'resnet50_XGBoost': {
                'model': xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss'),
                'extractor': 'resnet50'
            },
            'vgg16_RandomForest': {
                'model': RandomForestClassifier(n_estimators=100),
                'extractor': 'vgg16'
            },
            'densenet121_SVM': {
                'model': SVC(kernel='linear', probability=True),
                'extractor': 'densenet121'
            }
        },
        'scalers': {
            'resnet50': StandardScaler(),
            'vgg16': StandardScaler(),
            'densenet121': StandardScaler()
        },
        'is_demo': True
    }
    
    return demo_models


def predict_with_cnn(model, image_tensor):
    """Make prediction with CNN model."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.to(DEVICE))
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    return pred, confidence


def predict_with_shallow(model, features, scaler=None):
    """Make prediction with shallow classifier."""
    if scaler is not None:
        features = scaler.transform(features)
    
    pred = model.predict(features)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)[0]
        confidence = probs[pred]
    else:
        confidence = 0.5  # Default if no probability
    
    return int(pred), float(confidence)


def ensemble_predict(image_tensor, extractors, models_dict):
    """
    Make ensemble prediction using majority voting.
    
    Returns:
        dict with prediction, confidence, votes, and individual results
    """
    all_predictions = {}
    
    # 1. CNN predictions
    for name, cnn in models_dict.get('cnn_models', {}).items():
        try:
            pred, conf = predict_with_cnn(cnn, image_tensor)
            all_predictions[name] = {
                'prediction': pred,
                'confidence': conf,
                'class_name': 'Defective' if pred == 1 else 'Non-Defective'
            }
        except Exception as e:
            st.warning(f"Error with {name}: {e}")
    
    # 2. Shallow model predictions
    extracted_features = {}
    
    for name, model_info in models_dict.get('shallow_models', {}).items():
        try:
            extractor_name = model_info['extractor']
            
            # Extract features if not already done
            if extractor_name not in extracted_features:
                if extractor_name in extractors:
                    extractor = extractors[extractor_name]
                    extractor.eval()
                    with torch.no_grad():
                        features = extractor(image_tensor.to(DEVICE))
                        extracted_features[extractor_name] = features.cpu().numpy()
            
            if extractor_name in extracted_features:
                features = extracted_features[extractor_name]
                scaler = models_dict.get('scalers', {}).get(extractor_name)
                
                # For demo mode, simulate prediction
                if models_dict.get('is_demo', False):
                    # Simulate based on features (random but deterministic)
                    feature_sum = np.sum(features)
                    pred = 1 if (feature_sum % 2) > 0.5 else 0
                    conf = 0.6 + np.random.random() * 0.35
                else:
                    pred, conf = predict_with_shallow(model_info['model'], features, scaler)
                
                all_predictions[name] = {
                    'prediction': pred,
                    'confidence': conf,
                    'class_name': 'Defective' if pred == 1 else 'Non-Defective'
                }
        except Exception as e:
            st.warning(f"Error with {name}: {e}")
    
    # 3. Majority voting
    if not all_predictions:
        return {
            'prediction': 0,
            'class_name': 'Unknown',
            'confidence': 0.0,
            'num_models': 0,
            'votes': {'defective': 0, 'non_defective': 0},
            'model_results': {}
        }
    
    votes = [p['prediction'] for p in all_predictions.values()]
    defective_votes = sum(votes)
    non_defective_votes = len(votes) - defective_votes
    
    final_pred = 1 if defective_votes > non_defective_votes else 0
    confidence = max(defective_votes, non_defective_votes) / len(votes)
    
    return {
        'prediction': final_pred,
        'class_name': 'Defective' if final_pred == 1 else 'Non-Defective',
        'confidence': confidence,
        'num_models': len(all_predictions),
        'votes': {
            'defective': defective_votes,
            'non_defective': non_defective_votes
        },
        'model_results': all_predictions
    }


def create_gauge_chart(value, title="Confidence"):
    """Create a gauge chart for confidence visualization."""
    color = "#F44336" if value > 0.7 else "#FFC107" if value > 0.5 else "#4CAF50"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': '%', 'font': {'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#E8F5E9'},
                {'range': [50, 75], 'color': '#FFF3E0'},
                {'range': [75, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_vote_chart(votes):
    """Create a pie chart for vote distribution."""
    labels = ['Non-Defective', 'Defective']
    values = [votes['non_defective'], votes['defective']]
    colors = ['#4CAF50', '#F44336']
    
    fig = px.pie(
        values=values,
        names=labels,
        color_discrete_sequence=colors,
        hole=0.4
    )
    
    fig.update_layout(
        title="Vote Distribution",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2)
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='value+percent',
        textfont_size=14
    )
    
    return fig


# ============================================
# MAIN APP
# ============================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Détection de Défauts Industriels</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Upload an image to detect defects using ensemble AI models</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/100/factory.png", width=80)
        st.title("⚙️ Settings")
        
        st.markdown("---")
        
        # Mode selection
        use_demo = st.checkbox(
            "🧪 Demo Mode",
            value=True,
            help="Use simulated models for demonstration (no trained models required)"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Models")
        st.markdown("""
        **Ensemble includes:**
        - CNN Baseline
        - ResNet50 + SVM
        - ResNet50 + XGBoost
        - VGG16 + Random Forest
        - DenseNet121 + SVM
        """)
        
        st.markdown("---")
        
        # Device info
        st.subheader("💻 System")
        st.info(f"Device: {DEVICE}")
        
        st.markdown("---")
        st.markdown("**Hackathon 2024** 🚀")
    
    # Load models
    with st.spinner("Loading models..."):
        extractors = load_feature_extractors()
        
        if use_demo:
            models_dict = create_demo_models()
            st.info("🧪 **Demo Mode**: Using simulated predictions for UI demonstration.")
        else:
            models_dict = load_models()
            if not models_dict.get('cnn_models') and not models_dict.get('shallow_models'):
                st.warning("⚠️ No trained models found. Using demo mode.")
                models_dict = create_demo_models()
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of the industrial component to analyze"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            **File:** `{uploaded_file.name}`  
            **Size:** {image.size[0]} × {image.size[1]} px
            """)
    
    with col2:
        st.subheader("🔬 Analysis Results")
        
        if uploaded_file is not None:
            # Analyze button
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                
                # Preprocess image
                transform = get_transform()
                image_tensor = transform(image).unsqueeze(0)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Extracting features...")
                progress_bar.progress(20)
                time.sleep(0.3)
                
                status_text.text("Running ensemble prediction...")
                progress_bar.progress(50)
                
                # Make prediction
                start_time = time.time()
                result = ensemble_predict(image_tensor, extractors, models_dict)
                elapsed = time.time() - start_time
                
                progress_bar.progress(90)
                status_text.text("Generating results...")
                time.sleep(0.2)
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                # Display result
                st.markdown("---")
                
                if result['class_name'] == 'Defective':
                    st.markdown(f"""
                    <div class="result-defective">
                        <h2>⚠️ DEFECTIVE</h2>
                        <p>Confidence: <strong>{result['confidence']:.1%}</strong> ({result['votes']['defective']}/{result['num_models']} models)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-ok">
                        <h2>✅ NON-DEFECTIVE</h2>
                        <p>Confidence: <strong>{result['confidence']:.1%}</strong> ({result['votes']['non_defective']}/{result['num_models']} models)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed results in tabs
                tab1, tab2, tab3 = st.tabs(["📊 Vote Details", "🤖 Model Predictions", "📈 Confidence"])
                
                with tab1:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        # Vote distribution pie chart
                        if result['num_models'] > 0:
                            fig = create_vote_chart(result['votes'])
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_b:
                        st.markdown("### 🗳️ Voting Summary")
                        st.markdown(f"""
                        | Metric | Value |
                        |--------|-------|
                        | **Total Models** | {result['num_models']} |
                        | **Defective Votes** | {result['votes']['defective']} |
                        | **Non-Defective Votes** | {result['votes']['non_defective']} |
                        | **Final Decision** | {result['class_name']} |
                        | **Confidence** | {result['confidence']:.1%} |
                        | **Processing Time** | {elapsed:.2f}s |
                        """)
                
                with tab2:
                    st.markdown("### 🤖 Individual Model Predictions")
                    
                    if result['model_results']:
                        # Create DataFrame for display
                        model_data = []
                        for model_name, pred in result['model_results'].items():
                            model_data.append({
                                'Model': model_name,
                                'Prediction': pred['class_name'],
                                'Confidence': f"{pred['confidence']:.1%}"
                            })
                        
                        df = pd.DataFrame(model_data)
                        
                        # Style the dataframe
                        def highlight_prediction(val):
                            if val == 'Defective':
                                return 'background-color: #FFCDD2; color: #C62828'
                            else:
                                return 'background-color: #C8E6C9; color: #2E7D32'
                        
                        styled_df = df.style.applymap(
                            highlight_prediction, 
                            subset=['Prediction']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True, hide_index=True)
                        
                        # Visual representation
                        st.markdown("#### Visual Summary")
                        cols = st.columns(len(result['model_results']))
                        
                        for i, (model_name, pred) in enumerate(result['model_results'].items()):
                            with cols[i]:
                                icon = "🔴" if pred['prediction'] == 1 else "🟢"
                                short_name = model_name.replace('_', '\n')
                                st.markdown(f"""
                                <div style="text-align: center; padding: 10px; background: {'#FFEBEE' if pred['prediction'] == 1 else '#E8F5E9'}; border-radius: 10px;">
                                    <div style="font-size: 24px;">{icon}</div>
                                    <div style="font-size: 11px; font-weight: bold;">{short_name}</div>
                                    <div style="font-size: 10px;">{pred['confidence']:.0%}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No model predictions available.")
                
                with tab3:
                    # Confidence gauge
                    fig = create_gauge_chart(result['confidence'], "Ensemble Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if result['confidence'] >= 0.8:
                        st.success("✅ **High confidence** - The models strongly agree on this prediction.")
                    elif result['confidence'] >= 0.6:
                        st.warning("⚠️ **Medium confidence** - There is some disagreement between models.")
                    else:
                        st.error("❌ **Low confidence** - Models are split. Manual inspection recommended.")
        else:
            # Placeholder when no image uploaded
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px; border: 2px dashed #ccc;">
                <h3 style="color: #999;">👈 Upload an image to start</h3>
                <p style="color: #bbb;">Supported formats: JPG, JPEG, PNG, BMP</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 12px;">
        🏭 Industrial Defect Detection System | Hackathon 2024 | 
        Ensemble: CNN + ResNet50 + VGG16 + DenseNet121
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
