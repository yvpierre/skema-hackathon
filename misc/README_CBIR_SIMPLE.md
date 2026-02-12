# 🔍 CBIR Simplifié - Guide des Candidats

## 📁 Fichiers Fournis

| Fichier | Description |
|---------|-------------|
| `create_signatures.py` | Script pour créer les bases de signatures CBIR |
| `streamlit_app_complete.py` | Application Streamlit avec Classification + CBIR |

---

## 🚀 Étape 1: Créer les Signatures

### 1.1 Ouvrir `create_signatures.py`

### 1.2 Modifier la section CONFIGURATION

```python
# ╔══════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝

# Chemin vers le dossier des images d'entraînement
DATA_DIR = "./data/train"                    # ← Modifier ici

# Dossier de sortie pour les signatures
OUTPUT_DIR = "./signatures"                  # ← Modifier ici

# Modèles à utiliser
MODELS_TO_USE = ['resnet50', 'vgg16', 'densenet121']  # ← Modifier ici

# Normaliser les features (recommandé)
NORMALIZE_FEATURES = True
```

### 1.3 Exécuter le script

```bash
python create_signatures.py
```

### 1.4 Résultat

```
signatures/
├── signatures_resnet50.pkl
├── signatures_vgg16.pkl
└── signatures_densenet121.pkl
```

---

## 🚀 Étape 2: Lancer l'Application Streamlit

### 2.1 Ouvrir `streamlit_app_complete.py`

### 2.2 Modifier la section CONFIGURATION

```python
# ╔══════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝

# Dossier des modèles de classification
MODELS_DIR = Path("./models")                # ← Modifier ici

# Dossier des signatures CBIR
SIGNATURES_DIR = Path("./signatures")        # ← Modifier ici
```

### 2.3 Lancer l'application

```bash
streamlit run streamlit_app_complete.py
```

### 2.4 Ouvrir dans le navigateur

```
http://localhost:8501
```

---

## 📂 Structure des Dossiers Attendue

```
projet/
├── data/
│   ├── train/
│   │   ├── defective/
│   │   │   └── *.jpg
│   │   └── non_defective/
│   │       └── *.jpg
│   └── test/
│       ├── defective/
│       └── non_defective/
├── models/                          # Modèles de classification
│   ├── baseline_cnn.pth
│   ├── resnet50_svm.pkl
│   ├── resnet50_scaler.pkl
│   └── ...
├── signatures/                      # Signatures CBIR
│   ├── signatures_resnet50.pkl
│   ├── signatures_vgg16.pkl
│   └── signatures_densenet121.pkl
├── create_signatures.py
└── streamlit_app_complete.py
```

---

## 📏 Métriques de Distance Disponibles

| Distance | Description | Quand l'utiliser |
|----------|-------------|------------------|
| **Euclidienne** | Distance en ligne droite | Usage général |
| **Manhattan** | Somme des différences absolues | Robuste aux outliers |
| **Cosinus** | Angle entre vecteurs | **⭐ Recommandée** |
| **Chebyshev** | Maximum des différences | Une feature décisive |
| **Canberra** | Version pondérée | Magnitudes variées |

---

## 🧠 Modèles Extracteurs Disponibles

| Modèle | Dimensions | Taille | Vitesse |
|--------|------------|--------|---------|
| `resnet50` | 2048 | Moyen | ⭐⭐⭐ |
| `vgg16` | 25088 | Lourd | ⭐⭐ |
| `densenet121` | 1024 | Léger | ⭐⭐⭐ |
| `mobilenet_v2` | 1280 | Ultra-léger | ⭐⭐⭐⭐ |
| `efficientnet_b0` | 1280 | Léger | ⭐⭐⭐⭐ |

---

## 🎯 Fonctionnalités de l'Application Streamlit

### 1. Classification par Vote Majoritaire
- Combine CNN Baseline + Shallow Classifiers (SVM, XGBoost, RF)
- Affiche la prédiction finale avec confiance
- Visualise la distribution des votes

### 2. CBIR - Recherche d'Images Similaires
- Recherche les K images les plus similaires
- Compare plusieurs extracteurs
- Affiche les distances et classes des résultats

### 3. Mode Démo
- Fonctionne sans modèles entraînés
- Simule les résultats pour tester l'interface

---

## ❓ Dépannage

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "No images found"
Vérifiez la structure des dossiers:
```
data/train/defective/*.jpg
data/train/non_defective/*.jpg
```

### "Signatures non disponibles"
Exécutez d'abord:
```bash
python create_signatures.py
```

### L'application est lente
- Utilisez `mobilenet_v2` au lieu de `vgg16`
- Réduisez le nombre de modèles dans `MODELS_TO_USE`

---

## 📝 Exemple de Code pour le Notebook

### Créer une signature manuellement

```python
import pickle
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Charger le modèle
model = models.resnet50(weights='IMAGENET1K_V1')
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# Transformer l'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Extraire les features
image = Image.open("image.jpg").convert('RGB')
tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    features = model(tensor).numpy().flatten()

print(f"Features shape: {features.shape}")  # (2048,)
```

### Recherche manuelle

```python
from scipy.spatial.distance import cdist

# Charger la base
with open("signatures/signatures_resnet50.pkl", "rb") as f:
    db = pickle.load(f)

# Features de la query (extraites comme ci-dessus)
query = features.reshape(1, -1)

# Calculer les distances (cosinus)
distances = cdist(query, db['features'], metric='cosine').flatten()

# Trier et afficher les 5 plus proches
top5_idx = np.argsort(distances)[:5]

for i, idx in enumerate(top5_idx):
    print(f"#{i+1}: {db['paths'][idx]}")
    print(f"    Distance: {distances[idx]:.4f}")
    print(f"    Classe: {db['labels'][idx]}")
```

---

**Bon Hackathon! 🚀**
