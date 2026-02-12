
"""
🏭 Application Complète - Prédiction + CBIR
============================================
Application Streamlit pour:
1. Prédiction de défauts avec vote majoritaire (CNN + Shallow classifiers)
2. Recherche d'images similaires (CBIR)

Usage:
    streamlit run streamlit_app_complete.py

Prérequis:
    pip install streamlit torch torchvision scikit-learn pillow plotly pandas scipy
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

# Classes
CLASSES = ['non_defective', 'defective']

# Paramètres
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           PAGE CONFIG                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="🏭 Détection de Défauts + CBIR",
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
        margin-bottom: 1rem;
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
    .cbir-result {
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .cbir-defective {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .cbir-ok {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
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
# ║                           FONCTIONS DE DISTANCE                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def euclidean_distance(query, database):
    """Distance Euclidienne: √Σ(xi - yi)²"""
    return cdist(query, database, metric='euclidean').flatten()

def manhattan_distance(query, database):
    """Distance Manhattan: Σ|xi - yi|"""
    return cdist(query, database, metric='cityblock').flatten()

def cosine_distance(query, database):
    """Distance Cosinus: 1 - cos(θ)"""
    return cdist(query, database, metric='cosine').flatten()

def chebyshev_distance(query, database):
    """Distance Chebyshev: max|xi - yi|"""
    return cdist(query, database, metric='chebyshev').flatten()

def canberra_distance(query, database):
    """Distance Canberra: Σ(|xi-yi|/(|xi|+|yi|))"""
    return cdist(query, database, metric='canberra').flatten()

DISTANCE_FUNCTIONS = {
    'Euclidienne': euclidean_distance,
    'Manhattan': manhattan_distance,
    'Cosinus': cosine_distance,
    'Chebyshev': chebyshev_distance,
    'Canberra': canberra_distance,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FONCTIONS UTILITAIRES                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def get_transform():
    """Transformation des images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


@st.cache_resource
def load_feature_extractors():
    """Charge les extracteurs de features."""
    extractors = {}
    for name in ['resnet50', 'vgg16', 'densenet121']:
        try:
            extractor = FeatureExtractor(name)
            extractor.eval()
            extractor.to(DEVICE)
            extractors[name] = extractor
        except Exception as e:
            st.warning(f"Impossible de charger {name}: {e}")
    return extractors


@st.cache_resource
def load_classification_models():
    """Charge les modèles de classification."""
    models_dict = {
        'cnn_models': {},
        'shallow_models': {},
        'scalers': {}
    }
    
    if not MODELS_DIR.exists():
        return models_dict
    
    # CNN Baseline
    cnn_path = MODELS_DIR / 'baseline_cnn.pth'
    if cnn_path.exists():
        try:
            cnn = BaselineCNN()
            cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
            cnn.eval()
            cnn.to(DEVICE)
            models_dict['cnn_models']['CNN_Baseline'] = cnn
        except Exception as e:
            pass
    
    # Shallow classifiers
    for pkl_file in MODELS_DIR.glob('*.pkl'):
        if '_scaler' in pkl_file.stem:
            # C'est un scaler
            name = pkl_file.stem.replace('_scaler', '')
            try:
                models_dict['scalers'][name] = joblib.load(pkl_file)
            except:
                pass
        else:
            # C'est un classifier
            try:
                model = joblib.load(pkl_file)
                # Extraire le nom de l'extracteur (ex: resnet50_svm -> resnet50)
                parts = pkl_file.stem.split('_')
                extractor_name = parts[0]
                models_dict['shallow_models'][pkl_file.stem] = {
                    'model': model,
                    'extractor': extractor_name
                }
            except:
                pass
    
    return models_dict


@st.cache_resource
def load_cbir_signatures():
    """Charge les bases de signatures CBIR."""
    signatures = {}
    
    if not SIGNATURES_DIR.exists():
        return signatures
    
    for pkl_file in SIGNATURES_DIR.glob('signatures_*.pkl'):
        try:
            with open(pkl_file, 'rb') as f:
                db = pickle.load(f)
            model_name = db['metadata']['model_name']
            signatures[model_name] = db
        except Exception as e:
            st.warning(f"Erreur avec {pkl_file}: {e}")
    
    return signatures


def predict_ensemble(image_tensor, extractors, models_dict):
    """Prédiction avec vote majoritaire."""
    predictions = {}
    
    # CNN predictions
    for name, cnn in models_dict.get('cnn_models', {}).items():
        try:
            cnn.eval()
            with torch.no_grad():
                output = cnn(image_tensor.to(DEVICE))
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = probs[0][pred].item()
            predictions[name] = {'prediction': pred, 'confidence': conf}
        except:
            pass
    
    # Shallow predictions
    extracted_features = {}
    for name, info in models_dict.get('shallow_models', {}).items():
        try:
            ext_name = info['extractor']
            
            if ext_name not in extracted_features and ext_name in extractors:
                extractor = extractors[ext_name]
                with torch.no_grad():
                    feats = extractor(image_tensor.to(DEVICE))
                    extracted_features[ext_name] = feats.cpu().numpy()
            
            if ext_name in extracted_features:
                features = extracted_features[ext_name]
                scaler = models_dict.get('scalers', {}).get(ext_name)
                if scaler:
                    features = scaler.transform(features)
                
                pred = info['model'].predict(features)[0]
                conf = 0.8  # Default confidence
                if hasattr(info['model'], 'predict_proba'):
                    probs = info['model'].predict_proba(features)[0]
                    conf = probs[pred]
                
                predictions[name] = {'prediction': int(pred), 'confidence': float(conf)}
        except:
            pass
    
    if not predictions:
        return None
    
    # Vote majoritaire
    votes = [p['prediction'] for p in predictions.values()]
    defective_votes = sum(votes)
    total = len(votes)
    
    final_pred = 1 if defective_votes > total / 2 else 0
    confidence = max(defective_votes, total - defective_votes) / total
    
    return {
        'prediction': final_pred,
        'class_name': CLASSES[final_pred],
        'confidence': confidence,
        'votes': {'defective': defective_votes, 'non_defective': total - defective_votes},
        'model_results': predictions
    }


def cbir_search(image_tensor, extractor, signature_db, k=5, distance_metric='Cosinus'):
    """Recherche CBIR."""
    
    # Extraire les features de la query
    extractor.eval()
    with torch.no_grad():
        query_features = extractor(image_tensor.to(DEVICE))
        query_features = query_features.cpu().numpy()
    
    # Normaliser si la base est normalisée
    if signature_db['metadata'].get('normalized', False):
        norm = np.linalg.norm(query_features)
        if norm > 0:
            query_features = query_features / norm
    
    # Calculer les distances
    distance_func = DISTANCE_FUNCTIONS[distance_metric]
    distances = distance_func(query_features, signature_db['features'])
    
    # Trier et prendre les K premiers
    sorted_indices = np.argsort(distances)[:k]
    
    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        results.append({
            'rank': rank,
            'path': signature_db['paths'][idx],
            'distance': float(distances[idx]),
            'label': int(signature_db['labels'][idx]),
            'class_name': CLASSES[signature_db['labels'][idx]]
        })
    
    return results


def create_gauge_chart(value, title="Confiance"):
    """Crée un gauge pour la confiance."""
    color = "#F44336" if value > 0.7 else "#FFC107" if value > 0.5 else "#4CAF50"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#E8F5E9'},
                {'range': [50, 75], 'color': '#FFF3E0'},
                {'range': [75, 100], 'color': '#FFEBEE'}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           APPLICATION PRINCIPALE                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Détection de Défauts Industriels</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Classification par vote majoritaire + Recherche d\'images similaires (CBIR)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        st.markdown("---")
        
        # Mode demo
        demo_mode = st.checkbox("🧪 Mode Démo", value=True, 
                               help="Simule les résultats si aucun modèle n'est chargé")
        
        st.markdown("---")
        
        # Paramètres CBIR
        st.subheader("🔍 Paramètres CBIR")
        
        k_results = st.slider("Nombre de résultats (K)", 1, 10, 5)
        
        distance_metric = st.selectbox(
            "Métrique de distance",
            list(DISTANCE_FUNCTIONS.keys()),
            index=2  # Cosinus par défaut
        )
        
        st.markdown("---")
        st.info(f"💻 Device: {DEVICE}")
    
    # Charger les ressources
    with st.spinner("Chargement des modèles..."):
        extractors = load_feature_extractors()
        models_dict = load_classification_models()
        signatures = load_cbir_signatures()
    
    # Afficher le statut
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        n_classifiers = len(models_dict.get('cnn_models', {})) + len(models_dict.get('shallow_models', {}))
        st.metric("Classifiers", n_classifiers if n_classifiers > 0 else "Demo")
    with col_status2:
        st.metric("Extracteurs", len(extractors))
    with col_status3:
        st.metric("Signatures CBIR", len(signatures) if signatures else "Non disponible")
    
    st.markdown("---")
    
    # Upload
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image uploadée", use_container_width=True)
    
    with col2:
        st.subheader("🔬 Résultats d'Analyse")
        
        if uploaded_file:
            if st.button("🚀 Analyser", type="primary", use_container_width=True):
                
                # Préparer l'image
                transform = get_transform()
                image_tensor = transform(image).unsqueeze(0)
                
                # Progress
                progress = st.progress(0)
                status = st.empty()
                
                # =============================================
                # PARTIE 1: CLASSIFICATION
                # =============================================
                status.text("🎯 Classification en cours...")
                progress.progress(30)
                
                if demo_mode or not models_dict.get('cnn_models'):
                    # Mode démo: simuler
                    time.sleep(0.5)
                    pred_result = {
                        'prediction': np.random.choice([0, 1]),
                        'class_name': np.random.choice(['non_defective', 'defective']),
                        'confidence': np.random.uniform(0.6, 0.95),
                        'votes': {'defective': np.random.randint(2, 5), 'non_defective': np.random.randint(1, 3)},
                        'model_results': {
                            'CNN_Baseline': {'prediction': 1, 'confidence': 0.85},
                            'ResNet50_SVM': {'prediction': 1, 'confidence': 0.90},
                            'VGG16_RF': {'prediction': 0, 'confidence': 0.75},
                        }
                    }
                    pred_result['votes'] = {
                        'defective': sum(1 for m in pred_result['model_results'].values() if m['prediction'] == 1),
                        'non_defective': sum(1 for m in pred_result['model_results'].values() if m['prediction'] == 0)
                    }
                    pred_result['prediction'] = 1 if pred_result['votes']['defective'] > pred_result['votes']['non_defective'] else 0
                    pred_result['class_name'] = CLASSES[pred_result['prediction']]
                    pred_result['confidence'] = max(pred_result['votes'].values()) / sum(pred_result['votes'].values())
                else:
                    pred_result = predict_ensemble(image_tensor, extractors, models_dict)
                
                progress.progress(60)
                
                # =============================================
                # PARTIE 2: CBIR
                # =============================================
                status.text("🔍 Recherche d'images similaires...")
                
                cbir_results = {}
                
                if signatures:
                    for model_name, sig_db in signatures.items():
                        if model_name in extractors:
                            try:
                                results = cbir_search(
                                    image_tensor, 
                                    extractors[model_name], 
                                    sig_db, 
                                    k=k_results,
                                    distance_metric=distance_metric
                                )
                                cbir_results[model_name] = results
                            except Exception as e:
                                st.warning(f"Erreur CBIR {model_name}: {e}")
                elif demo_mode:
                    # Simuler des résultats CBIR
                    for model_name in ['resnet50', 'vgg16']:
                        cbir_results[model_name] = [
                            {
                                'rank': i+1,
                                'path': f'./data/train/{"defective" if np.random.random() > 0.5 else "non_defective"}/img_{np.random.randint(1, 100):03d}.jpg',
                                'distance': np.random.uniform(0.1, 0.5),
                                'label': np.random.choice([0, 1]),
                                'class_name': np.random.choice(['non_defective', 'defective'])
                            }
                            for i in range(k_results)
                        ]
                
                progress.progress(100)
                status.empty()
                progress.empty()
                
                # =============================================
                # AFFICHAGE DES RÉSULTATS
                # =============================================
                
                st.markdown("---")
                
                # Résultat principal
                if pred_result:
                    if pred_result['prediction'] == 1:
                        st.markdown(f"""
                        <div class="result-defective">
                            <h2>⚠️ DÉFAUT DÉTECTÉ</h2>
                            <p>Confiance: <strong>{pred_result['confidence']:.1%}</strong> 
                            ({pred_result['votes']['defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-ok">
                            <h2>✅ PIÈCE CONFORME</h2>
                            <p>Confiance: <strong>{pred_result['confidence']:.1%}</strong>
                            ({pred_result['votes']['non_defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Tabs pour détails
                tab1, tab2, tab3 = st.tabs(["📊 Vote Majoritaire", "🔍 CBIR - Images Similaires", "📈 Détails"])
                
                with tab1:
                    if pred_result and pred_result.get('model_results'):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # Pie chart des votes
                            fig = px.pie(
                                values=[pred_result['votes']['non_defective'], pred_result['votes']['defective']],
                                names=['Non-Défectueux', 'Défectueux'],
                                color_discrete_sequence=['#4CAF50', '#F44336'],
                                hole=0.4
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            # Tableau des modèles
                            model_data = []
                            for name, res in pred_result['model_results'].items():
                                model_data.append({
                                    'Modèle': name,
                                    'Prédiction': '🔴 Défaut' if res['prediction'] == 1 else '🟢 OK',
                                    'Confiance': f"{res['confidence']:.1%}"
                                })
                            st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
                
                with tab2:
                    if cbir_results:
                        st.markdown(f"**Métrique:** {distance_metric} | **K:** {k_results}")
                        
                        for model_name, results in cbir_results.items():
                            st.markdown(f"### 🧠 {model_name.upper()}")
                            
                            cols = st.columns(min(k_results, 5))
                            
                            for i, res in enumerate(results[:5]):
                                with cols[i]:
                                    # Essayer de charger l'image
                                    try:
                                        if Path(res['path']).exists():
                                            img = Image.open(res['path'])
                                            st.image(img, use_container_width=True)
                                        else:
                                            st.info(f"📷 path: {res['path']} #{res['rank']}")
                                    except:
                                        st.info(f"📷 Image #{res['rank']}")
                                    
                                    # Afficher les infos
                                    status_icon = "🔴" if res['label'] == 1 else "🟢"
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 5px; 
                                         background: {'#FFEBEE' if res['label'] == 1 else '#E8F5E9'}; 
                                         border-radius: 5px;">
                                        <b>#{res['rank']}</b> {status_icon}<br>
                                        <small>Dist: {res['distance']:.4f}</small>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            st.markdown("---")
                    else:
                        st.info("Aucune base de signatures CBIR disponible. Créez-en avec `create_signatures.py`")
                
                with tab3:
                    if pred_result:
                        # Gauge de confiance
                        fig = create_gauge_chart(pred_result['confidence'], "Confiance Ensemble")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Interprétation
                        if pred_result['confidence'] >= 0.8:
                            st.success("✅ **Haute confiance** - Les modèles sont d'accord")
                        elif pred_result['confidence'] >= 0.6:
                            st.warning("⚠️ **Confiance moyenne** - Désaccord entre modèles")
                        else:
                            st.error("❌ **Faible confiance** - Vérification manuelle recommandée")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f5f5f5; border-radius: 10px; border: 2px dashed #ccc;">
                <h3 style="color: #999;">👈 Uploadez une image pour commencer</h3>
                <p style="color: #bbb;">Formats: JPG, JPEG, PNG, BMP</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 12px;">
        🏭 Hackathon IA - Détection de Défauts Industriels | Classification + CBIR
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
