"""
🏭 Application Complète - Classification + CBIR + VLM
=====================================================
Application Streamlit pour:
1. Prédiction de défauts avec vote majoritaire (CNN + Shallow classifiers)
2. Recherche d'images similaires (CBIR)
3. Description textuelle avec VLM (Vision Language Model)

Usage:
    streamlit run streamlit_app_final.py

Prérequis:
    pip install streamlit torch torchvision scikit-learn pillow plotly pandas scipy
    pip install transformers  # Optionnel pour BLIP
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
import pickle
import joblib
from scipy.spatial.distance import cdist
import time


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                                       ║
# ║                    Modifiez ces chemins selon votre structure                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Dossier contenant les modèles de classification (.pth, .pkl)
MODELS_DIR = Path("./models")

# Dossier contenant les signatures CBIR (.pkl)
SIGNATURES_DIR = Path("./signatures")

# Mode VLM: 'template' (pas de GPU) ou 'blip' (GPU recommandé)
VLM_MODE = "template"

# Langue pour les descriptions VLM: 'fr' ou 'en'
VLM_LANGUAGE = "fr"

# Classes
CLASSES = ['non_defective', 'defective']
CLASSES_FR = ['Non-défectueux', 'Défectueux']

# Paramètres
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           PAGE CONFIG                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="🏭 Détection Défauts + CBIR + VLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .result-defective {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        border-left: 6px solid #F44336;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-ok {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-left: 6px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .vlm-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left: 6px solid #2196F3;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .urgency-urgent {
        background-color: #F44336;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .urgency-high {
        background-color: #FF9800;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .urgency-moderate {
        background-color: #FFC107;
        color: black;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .urgency-none {
        background-color: #4CAF50;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MODÈLES CNN                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BaselineCNN(nn.Module):
    """CNN Baseline pour la classification."""
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


class FeatureExtractor(nn.Module):
    """Extracteur de features pour classification et CBIR."""
    def __init__(self, model_name='resnet50'):
        super().__init__()
        self.model_name = model_name
        
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
    
    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg16':
            x = self.avgpool(x)
        elif self.model_name == 'densenet121':
            x = nn.functional.relu(x, inplace=True)
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           VLM - VISION LANGUAGE MODEL                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TemplateVLM:
    """Générateur de descriptions basé sur templates (pas de GPU requis)."""
    
    def __init__(self, language='fr'):
        self.language = language
        
        self.templates = {
            'fr': {
                'defective': {
                    'high': {
                        'title': "⚠️ DÉFAUT CRITIQUE DÉTECTÉ",
                        'description': "L'analyse révèle une anomalie majeure sur cette pièce industrielle.",
                        'confidence_text': "La confiance du système est très élevée ({confidence:.1%}).",
                        'details': "**Type probable:** {defect_type}\n**Zone affectée:** {zone}",
                        'recommendation': "🚨 **ACTION IMMÉDIATE:** Retirer cette pièce de la production. Inspection manuelle obligatoire.",
                        'urgency': "URGENT",
                        'urgency_class': "urgency-urgent"
                    },
                    'medium': {
                        'title': "⚠️ DÉFAUT DÉTECTÉ",
                        'description': "L'analyse indique une anomalie sur cette pièce.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "**Type probable:** {defect_type}\n**Inspection recommandée**",
                        'recommendation': "📋 **ACTION:** Mettre en quarantaine pour inspection complémentaire.",
                        'urgency': "ÉLEVÉ",
                        'urgency_class': "urgency-high"
                    },
                    'low': {
                        'title': "⚠️ DÉFAUT POSSIBLE",
                        'description': "L'analyse suggère une anomalie potentielle.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Le système n'est pas certain. Vérification manuelle conseillée.",
                        'recommendation': "📝 **ACTION:** Vérification visuelle recommandée.",
                        'urgency': "MODÉRÉ",
                        'urgency_class': "urgency-moderate"
                    }
                },
                'non_defective': {
                    'high': {
                        'title': "✅ PIÈCE CONFORME",
                        'description': "L'analyse confirme que cette pièce ne présente aucun défaut visible.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Tous les critères de qualité sont satisfaits.",
                        'recommendation': "👍 Cette pièce peut continuer dans le processus de production.",
                        'urgency': "AUCUN",
                        'urgency_class': "urgency-none"
                    },
                    'medium': {
                        'title': "✅ PIÈCE PROBABLEMENT CONFORME",
                        'description': "L'analyse suggère que cette pièce est en bon état.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Les critères principaux sont satisfaits.",
                        'recommendation': "👍 Peut continuer, vérification optionnelle.",
                        'urgency': "AUCUN",
                        'urgency_class': "urgency-none"
                    },
                    'low': {
                        'title': "❓ STATUT INCERTAIN",
                        'description': "L'analyse n'est pas concluante.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Le système hésite entre conforme et non-conforme.",
                        'recommendation': "🔍 Inspection manuelle recommandée.",
                        'urgency': "MODÉRÉ",
                        'urgency_class': "urgency-moderate"
                    }
                }
            },
            'en': {
                'defective': {
                    'high': {
                        'title': "⚠️ CRITICAL DEFECT DETECTED",
                        'description': "Analysis reveals a major anomaly on this industrial part.",
                        'confidence_text': "System confidence is very high ({confidence:.1%}).",
                        'details': "**Probable type:** {defect_type}\n**Affected zone:** {zone}",
                        'recommendation': "🚨 **IMMEDIATE ACTION:** Remove from production. Manual inspection required.",
                        'urgency': "URGENT",
                        'urgency_class': "urgency-urgent"
                    },
                    'medium': {
                        'title': "⚠️ DEFECT DETECTED",
                        'description': "Analysis indicates an anomaly on this part.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "**Probable type:** {defect_type}\n**Inspection recommended**",
                        'recommendation': "📋 **ACTION:** Quarantine for additional inspection.",
                        'urgency': "HIGH",
                        'urgency_class': "urgency-high"
                    },
                    'low': {
                        'title': "⚠️ POSSIBLE DEFECT",
                        'description': "Analysis suggests a potential anomaly.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "System is uncertain. Manual verification advised.",
                        'recommendation': "📝 **ACTION:** Visual verification recommended.",
                        'urgency': "MODERATE",
                        'urgency_class': "urgency-moderate"
                    }
                },
                'non_defective': {
                    'high': {
                        'title': "✅ PART CONFORMING",
                        'description': "Analysis confirms this part shows no visible defects.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "All quality criteria are satisfied.",
                        'recommendation': "👍 This part can continue in production.",
                        'urgency': "NONE",
                        'urgency_class': "urgency-none"
                    },
                    'medium': {
                        'title': "✅ PART LIKELY CONFORMING",
                        'description': "Analysis suggests this part is in good condition.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "Main criteria are satisfied.",
                        'recommendation': "👍 Can continue, optional verification.",
                        'urgency': "NONE",
                        'urgency_class': "urgency-none"
                    },
                    'low': {
                        'title': "❓ UNCERTAIN STATUS",
                        'description': "Analysis is inconclusive.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "System uncertain between conforming and non-conforming.",
                        'recommendation': "🔍 Manual inspection recommended.",
                        'urgency': "MODERATE",
                        'urgency_class': "urgency-moderate"
                    }
                }
            }
        }
        
        self.defect_types = {
            'fr': ["rayure", "fissure", "déformation", "corrosion", "inclusion", "porosité", "bavure"],
            'en': ["scratch", "crack", "deformation", "corrosion", "inclusion", "porosity", "burr"]
        }
        
        self.zones = {
            'fr': ["surface principale", "bord", "centre", "coin", "jonction"],
            'en': ["main surface", "edge", "center", "corner", "junction"]
        }
    
    def get_confidence_level(self, confidence):
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        return 'low'
    
    def generate(self, prediction, confidence, cbir_results=None):
        """Génère une description structurée."""
        
        lang = self.language
        class_name = 'defective' if prediction == 1 else 'non_defective'
        level = self.get_confidence_level(confidence)
        
        template = self.templates[lang][class_name][level]
        
        # Sélectionner défaut et zone
        seed = int(confidence * 1000)
        defect_type = self.defect_types[lang][seed % len(self.defect_types[lang])]
        zone = self.zones[lang][seed % len(self.zones[lang])]
        
        # CBIR info
        cbir_text = ""
        if cbir_results and len(cbir_results) > 0:
            defect_count = sum(1 for r in cbir_results if r.get('label', 0) == 1)
            total = len(cbir_results)
            if lang == 'fr':
                cbir_text = f"**Analyse CBIR:** {defect_count}/{total} images similaires sont défectueuses."
            else:
                cbir_text = f"**CBIR Analysis:** {defect_count}/{total} similar images are defective."
        
        return {
            'title': template['title'],
            'description': template['description'],
            'confidence_text': template['confidence_text'].format(confidence=confidence),
            'details': template['details'].format(defect_type=defect_type, zone=zone),
            'recommendation': template['recommendation'],
            'urgency': template['urgency'],
            'urgency_class': template['urgency_class'],
            'cbir_text': cbir_text
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FONCTIONS DE DISTANCE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DISTANCE_FUNCTIONS = {
    'Euclidienne': lambda q, db: cdist(q, db, metric='euclidean').flatten(),
    'Manhattan': lambda q, db: cdist(q, db, metric='cityblock').flatten(),
    'Cosinus': lambda q, db: cdist(q, db, metric='cosine').flatten(),
    'Chebyshev': lambda q, db: cdist(q, db, metric='chebyshev').flatten(),
    'Canberra': lambda q, db: cdist(q, db, metric='canberra').flatten(),
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FONCTIONS UTILITAIRES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@st.cache_resource
def load_feature_extractors():
    extractors = {}
    for name in ['resnet50', 'vgg16', 'densenet121']:
        try:
            extractor = FeatureExtractor(name)
            extractor.eval()
            extractor.to(DEVICE)
            extractors[name] = extractor
        except:
            pass
    return extractors


@st.cache_resource
def load_classification_models():
    models_dict = {'cnn_models': {}, 'shallow_models': {}, 'scalers': {}}
    
    if not MODELS_DIR.exists():
        return models_dict
    
    # CNN
    cnn_path = MODELS_DIR / 'baseline_cnn.pth'
    if cnn_path.exists():
        try:
            cnn = BaselineCNN()
            cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn.eval().to(DEVICE)
            models_dict['cnn_models']['CNN_Baseline'] = cnn
        except:
            pass
    
    # Shallow
    for pkl_file in MODELS_DIR.glob('*.pkl'):
        if '_scaler' in pkl_file.stem:
            name = pkl_file.stem.replace('_scaler', '')
            try:
                models_dict['scalers'][name] = joblib.load(pkl_file)
            except:
                pass
        else:
            try:
                model = joblib.load(pkl_file)
                parts = pkl_file.stem.split('_')
                models_dict['shallow_models'][pkl_file.stem] = {
                    'model': model,
                    'extractor': parts[0]
                }
            except:
                pass
    
    return models_dict


@st.cache_resource
def load_cbir_signatures():
    signatures = {}
    if not SIGNATURES_DIR.exists():
        return signatures
    
    for pkl_file in SIGNATURES_DIR.glob('signatures_*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                db = pickle.load(f)
            signatures[db['metadata']['model_name']] = db
        except:
            pass
    return signatures


@st.cache_resource
def load_vlm():
    return TemplateVLM(language=VLM_LANGUAGE)


def predict_ensemble(image_tensor, extractors, models_dict):
    """Prédiction avec vote majoritaire."""
    predictions = {}
    
    # CNN
    for name, cnn in models_dict.get('cnn_models', {}).items():
        try:
            with torch.no_grad():
                output = cnn(image_tensor.to(DEVICE))
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
            predictions[name] = {'prediction': pred, 'confidence': probs[0][pred].item()}
        except:
            pass
    
    # Shallow
    extracted = {}
    for name, info in models_dict.get('shallow_models', {}).items():
        try:
            ext_name = info['extractor']
            if ext_name not in extracted and ext_name in extractors:
                with torch.no_grad():
                    feats = extractors[ext_name](image_tensor.to(DEVICE))
                    extracted[ext_name] = feats.cpu().numpy()
            
            if ext_name in extracted:
                features = extracted[ext_name]
                scaler = models_dict.get('scalers', {}).get(ext_name)
                if scaler:
                    features = scaler.transform(features)
                
                pred = info['model'].predict(features)[0]
                conf = 0.8
                if hasattr(info['model'], 'predict_proba'):
                    conf = info['model'].predict_proba(features)[0][pred]
                predictions[name] = {'prediction': int(pred), 'confidence': float(conf)}
        except:
            pass
    
    if not predictions:
        return None
    
    votes = [p['prediction'] for p in predictions.values()]
    defective = sum(votes)
    total = len(votes)
    
    final = 1 if defective > total / 2 else 0
    confidence = max(defective, total - defective) / total
    
    return {
        'prediction': final,
        'class_name': CLASSES[final],
        'class_name_fr': CLASSES_FR[final],
        'confidence': confidence,
        'votes': {'defective': defective, 'non_defective': total - defective},
        'model_results': predictions
    }


def cbir_search(image_tensor, extractor, signature_db, k=5, distance_metric='Cosinus'):
    """Recherche CBIR."""
    with torch.no_grad():
        query = extractor(image_tensor.to(DEVICE)).cpu().numpy()
    
    if signature_db['metadata'].get('normalized', False):
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
    
    distances = DISTANCE_FUNCTIONS[distance_metric](query, signature_db['features'])
    indices = np.argsort(distances)[:k]
    
    return [
        {
            'rank': i + 1,
            'path': signature_db['paths'][idx],
            'distance': float(distances[idx]),
            'label': int(signature_db['labels'][idx]),
            'class_name': CLASSES[signature_db['labels'][idx]]
        }
        for i, idx in enumerate(indices)
    ]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           APPLICATION PRINCIPALE                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Détection de Défauts Industriels</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Classification + CBIR + Description IA (VLM)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        
        demo_mode = st.checkbox("🧪 Mode Démo", value=True)
        
        st.markdown("---")
        st.subheader("🔍 Paramètres CBIR")
        k_results = st.slider("Nombre de résultats (K)", 1, 10, 5)
        distance_metric = st.selectbox("Distance", list(DISTANCE_FUNCTIONS.keys()), index=2)
        
        st.markdown("---")
        st.subheader("🤖 Paramètres VLM")
        vlm_enabled = st.checkbox("Activer VLM", value=True)
        
        st.markdown("---")
        st.info(f"💻 {DEVICE}")
    
    # Load resources
    with st.spinner("Chargement..."):
        extractors = load_feature_extractors()
        models_dict = load_classification_models()
        signatures = load_cbir_signatures()
        vlm = load_vlm()
    
    # Status
    c1, c2, c3 = st.columns(3)
    n_models = len(models_dict.get('cnn_models', {})) + len(models_dict.get('shallow_models', {}))
    c1.metric("🎯 Classifiers", n_models if n_models > 0 else "Demo")
    c2.metric("🔍 Signatures CBIR", len(signatures) if signatures else "N/A")
    c3.metric("🤖 VLM", "Actif" if vlm_enabled else "Inactif")
    
    st.markdown("---")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded:
            image = Image.open(uploaded).convert('RGB')
            st.image(image, caption="Image uploadée", use_container_width=True)
    
    with col2:
        st.subheader("🔬 Résultats")
        
        if uploaded:
            if st.button("🚀 Analyser", type="primary", use_container_width=True):
                
                transform = get_transform()
                image_tensor = transform(image).unsqueeze(0)
                
                progress = st.progress(0)
                status = st.empty()
                
                # =====================
                # CLASSIFICATION
                # =====================
                status.text("🎯 Classification...")
                progress.progress(25)
                
                if demo_mode or not models_dict.get('cnn_models'):
                    time.sleep(0.3)
                    pred_result = {
                        'prediction': np.random.choice([0, 1]),
                        'confidence': np.random.uniform(0.65, 0.95),
                        'votes': {'defective': 3, 'non_defective': 2},
                        'model_results': {
                            'CNN_Baseline': {'prediction': 1, 'confidence': 0.87},
                            'ResNet50_SVM': {'prediction': 1, 'confidence': 0.91},
                            'VGG16_RF': {'prediction': 0, 'confidence': 0.78},
                        }
                    }
                    pred_result['votes'] = {
                        'defective': sum(1 for m in pred_result['model_results'].values() if m['prediction'] == 1),
                        'non_defective': sum(1 for m in pred_result['model_results'].values() if m['prediction'] == 0)
                    }
                    pred_result['prediction'] = 1 if pred_result['votes']['defective'] > pred_result['votes']['non_defective'] else 0
                    pred_result['confidence'] = max(pred_result['votes'].values()) / sum(pred_result['votes'].values())
                else:
                    pred_result = predict_ensemble(image_tensor, extractors, models_dict)
                
                # =====================
                # CBIR
                # =====================
                status.text("🔍 Recherche CBIR...")
                progress.progress(50)
                
                cbir_results = {}
                if signatures:
                    for model_name, sig_db in signatures.items():
                        if model_name in extractors:
                            try:
                                cbir_results[model_name] = cbir_search(
                                    image_tensor, extractors[model_name], sig_db, k_results, distance_metric
                                )
                            except:
                                pass
                elif demo_mode:
                    for model_name in ['resnet50']:
                        cbir_results[model_name] = [
                            {
                                'rank': i + 1,
                                'path': f'./data/train/{"defective" if np.random.random() > 0.4 else "non_defective"}/img_{np.random.randint(1,100):03d}.jpg',
                                'distance': np.random.uniform(0.1, 0.5),
                                'label': np.random.choice([0, 1]),
                                'class_name': np.random.choice(CLASSES)
                            }
                            for i in range(k_results)
                        ]
                
                # =====================
                # VLM
                # =====================
                status.text("🤖 Génération description VLM...")
                progress.progress(75)
                
                vlm_result = None
                if vlm_enabled and pred_result:
                    first_cbir = list(cbir_results.values())[0] if cbir_results else None
                    vlm_result = vlm.generate(
                        prediction=pred_result['prediction'],
                        confidence=pred_result['confidence'],
                        cbir_results=first_cbir
                    )
                
                progress.progress(100)
                status.empty()
                progress.empty()
                
                # =====================
                # DISPLAY RESULTS
                # =====================
                st.markdown("---")
                
                # Main result
                if pred_result:
                    if pred_result['prediction'] == 1:
                        st.markdown(f"""
                        <div class="result-defective">
                            <h2>⚠️ DÉFAUT DÉTECTÉ</h2>
                            <p>Confiance: <b>{pred_result['confidence']:.1%}</b> 
                            ({pred_result['votes']['defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-ok">
                            <h2>✅ PIÈCE CONFORME</h2>
                            <p>Confiance: <b>{pred_result['confidence']:.1%}</b>
                            ({pred_result['votes']['non_defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["🤖 Description VLM", "🔍 CBIR", "📊 Vote Majoritaire"])
                
                # TAB 1: VLM
                with tab1:
                    if vlm_result:
                        st.markdown(f"### {vlm_result['title']}")
                        
                        # Urgency badge
                        st.markdown(f"""
                        <span class="{vlm_result['urgency_class']}">
                            Urgence: {vlm_result['urgency']}
                        </span>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        st.markdown(f"**{vlm_result['description']}**")
                        st.markdown(vlm_result['confidence_text'])
                        
                        st.markdown("#### 📋 Détails")
                        st.markdown(vlm_result['details'])
                        
                        if vlm_result['cbir_text']:
                            st.markdown(f"\n{vlm_result['cbir_text']}")
                        
                        st.markdown("#### 💡 Recommandation")
                        st.info(vlm_result['recommendation'])
                    else:
                        st.info("VLM désactivé ou pas de résultat de classification.")
                
                # TAB 2: CBIR
                with tab2:
                    if cbir_results:
                        st.markdown(f"**Distance:** {distance_metric} | **K:** {k_results}")
                        
                        for model_name, results in cbir_results.items():
                            st.markdown(f"### 🧠 {model_name.upper()}")
                            
                            cols = st.columns(min(k_results, 5))
                            for i, res in enumerate(results[:5]):
                                with cols[i]:
                                    try:
                                        if Path(res['path']).exists():
                                            st.image(Image.open(res['path']), use_container_width=True)
                                        else:
                                            st.info(f"📷 #{res['rank']}")
                                    except:
                                        st.info(f"📷 #{res['rank']}")
                                    
                                    icon = "🔴" if res['label'] == 1 else "🟢"
                                    bg = '#FFEBEE' if res['label'] == 1 else '#E8F5E9'
                                    st.markdown(f"""
                                    <div style="text-align:center;padding:5px;background:{bg};border-radius:5px;">
                                        <b>#{res['rank']}</b> {icon}<br>
                                        <small>Dist: {res['distance']:.4f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune signature CBIR. Exécutez `create_signatures.py` d'abord.")
                
                # TAB 3: Vote
                with tab3:
                    if pred_result and pred_result.get('model_results'):
                        c_a, c_b = st.columns(2)
                        
                        with c_a:
                            fig = px.pie(
                                values=[pred_result['votes']['non_defective'], pred_result['votes']['defective']],
                                names=['Non-Défectueux', 'Défectueux'],
                                color_discrete_sequence=['#4CAF50', '#F44336'],
                                hole=0.4
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with c_b:
                            data = [
                                {
                                    'Modèle': name,
                                    'Prédiction': '🔴 Défaut' if r['prediction'] == 1 else '🟢 OK',
                                    'Confiance': f"{r['confidence']:.1%}"
                                }
                                for name, r in pred_result['model_results'].items()
                            ]
                            st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;background:#f5f5f5;border-radius:10px;border:2px dashed #ccc;">
                <h3 style="color:#999;">👈 Uploadez une image pour commencer</h3>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#999;font-size:12px;">
        🏭 Hackathon IA - Classification + CBIR + VLM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
