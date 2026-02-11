"""
🔍 CBIR - Création des Bases de Signatures
==========================================
Script simplifié pour créer les bases de signatures CBIR.
Pas de paramètres en ligne de commande - modifiez directement les variables ci-dessous.

Usage:
    1. Modifiez les paramètres dans la section CONFIGURATION
    2. Exécutez: python create_signatures.py
"""

import os
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                                       ║
# ║                    Modifiez ces paramètres selon vos besoins                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Chemin vers le dossier des images d'entraînement
DATA_DIR = "./data/train"

# Dossier de sortie pour les signatures
OUTPUT_DIR = "./signatures"

# Modèles à utiliser pour créer les signatures
# Options: 'resnet50', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0'
MODELS_TO_USE = ['resnet50', 'vgg16', 'densenet121']

# Normaliser les features (recommandé pour distance cosinus)
NORMALIZE_FEATURES = True

# Noms des classes (dossiers)
CLASSES = ['non_defective', 'defective']

# Taille des images
IMG_SIZE = 224


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CODE (Ne pas modifier)                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"💻 Device: {DEVICE}")


class FeatureExtractor(nn.Module):
    """Extracteur de features CNN."""
    
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
            
        elif model_name == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.output_dim = 1280
            
        elif model_name == 'efficientnet_b0':
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = base.features
            self.output_dim = 1280
        
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        # Geler les poids
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg16':
            x = self.avgpool(x)
        elif self.model_name in ['densenet121', 'mobilenet_v2', 'efficientnet_b0']:
            x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)


def get_transform():
    """Transformation des images."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_images(data_dir):
    """Charge les chemins et labels des images."""
    data_dir = Path(data_dir)
    paths = []
    labels = []
    
    print(f"\n📂 Chargement depuis: {data_dir}")
    
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            print(f"   ⚠️ Dossier non trouvé: {class_dir}")
            continue
        
        count = 0
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
            for img_path in class_dir.glob(ext):
                paths.append(str(img_path))
                labels.append(label_idx)
                count += 1
        
        print(f"   ✅ {class_name}: {count} images")
    
    print(f"   📊 Total: {len(paths)} images")
    return paths, labels


def extract_features(image_paths, extractor, transform):
    """Extrait les features de toutes les images."""
    extractor = extractor.to(DEVICE)
    extractor.eval()
    
    all_features = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="   Extraction"):
            try:
                # Charger et transformer l'image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(DEVICE)
                
                # Extraire les features
                features = extractor(image_tensor)
                all_features.append(features.cpu().numpy())
                
            except Exception as e:
                print(f"   ❌ Erreur avec {img_path}: {e}")
                # Ajouter un vecteur de zéros pour garder l'alignement
                all_features.append(np.zeros((1, extractor.output_dim)))
    
    return np.vstack(all_features)


def create_signature_database(model_name, data_dir, output_dir, normalize=True):
    """Crée une base de signatures pour un modèle donné."""
    
    print(f"\n{'='*60}")
    print(f"🧠 {model_name.upper()}")
    print('='*60)
    
    # Charger les images
    paths, labels = load_images(data_dir)
    
    if len(paths) == 0:
        print("❌ Aucune image trouvée!")
        return None
    
    # Créer l'extracteur
    print(f"\n   Chargement du modèle {model_name}...")
    extractor = FeatureExtractor(model_name)
    print(f"   ✅ Modèle chargé (dim: {extractor.output_dim})")
    
    # Extraire les features
    print(f"\n   Extraction des features...")
    transform = get_transform()
    features = extract_features(paths, extractor, transform)
    print(f"   ✅ Features extraites: {features.shape}")
    
    # Normaliser
    if normalize:
        print(f"\n   Normalisation L2...")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        features = features / norms
        print(f"   ✅ Features normalisées")
    
    # Créer la base
    database = {
        'features': features,
        'paths': paths,
        'labels': np.array(labels),
        'metadata': {
            'model_name': model_name,
            'feature_dim': features.shape[1],
            'num_images': len(paths),
            'normalized': normalize,
            'created_at': datetime.now().isoformat(),
            'data_dir': str(data_dir),
            'class_distribution': {
                'non_defective': int(np.sum(np.array(labels) == 0)),
                'defective': int(np.sum(np.array(labels) == 1))
            }
        }
    }
    
    # Sauvegarder
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'signatures_{model_name}.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(database, f)
    
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"\n   💾 Sauvegardé: {output_path} ({file_size:.2f} MB)")
    
    return database


def main():
    """Fonction principale."""
    
    print("\n" + "="*60)
    print("🔍 CBIR - CRÉATION DES BASES DE SIGNATURES")
    print("="*60)
    
    print(f"\n📁 Dossier des données: {DATA_DIR}")
    print(f"📁 Dossier de sortie: {OUTPUT_DIR}")
    print(f"🧠 Modèles: {', '.join(MODELS_TO_USE)}")
    print(f"📏 Normalisation: {'Oui' if NORMALIZE_FEATURES else 'Non'}")
    
    # Vérifier que le dossier de données existe
    if not Path(DATA_DIR).exists():
        print(f"\n❌ ERREUR: Le dossier {DATA_DIR} n'existe pas!")
        print(f"\nStructure attendue:")
        print(f"   {DATA_DIR}/")
        print(f"   ├── defective/")
        print(f"   │   └── *.jpg")
        print(f"   └── non_defective/")
        print(f"       └── *.jpg")
        return
    
    # Créer les signatures pour chaque modèle
    created_databases = []
    
    for model_name in MODELS_TO_USE:
        db = create_signature_database(
            model_name=model_name,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            normalize=NORMALIZE_FEATURES
        )
        if db:
            created_databases.append(model_name)
    
    # Résumé
    print("\n" + "="*60)
    print("✅ CRÉATION TERMINÉE!")
    print("="*60)
    
    print(f"\n📁 Fichiers créés dans {OUTPUT_DIR}/:")
    for model_name in created_databases:
        print(f"   - signatures_{model_name}.pkl")
    
    print(f"\n🚀 Prochaine étape:")
    print(f"   Utilisez ces signatures dans votre application Streamlit")
    print(f"   pour la recherche d'images similaires (CBIR).")


if __name__ == '__main__':
    main()
