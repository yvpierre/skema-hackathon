# 🏭 SKEMA Hackathon - Système de Détection de Défauts Industriels

Un pipeline machine learning complet de bout en bout pour la détection automatisée de défauts industriels utilisant l'apprentissage par ensemble et la recherche d'images par le contenu (CBIR). Ce projet combine des techniques d'apprentissage profond, de machine learning classique et de vision par ordinateur pour classifier les composants industriels comme défectueux ou non-défectueux avec une grande confiance.

---
👉 **Démo en ligne :** [Application Streamlit](https://skema-hackathon.streamlit.app)

---

## 📋 Table des matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles](#modèles)
- [Données](#données)
- [Contribuer](#contribuer)

## 🎯 Aperçu

Ce système offre :
- **Prédiction par ensemble** : Combine plusieurs modèles de deep learning et ML classiques pour une classification robuste des défauts
- **Interface web interactive** : Application basée sur Streamlit pour un téléversement et une analyse faciles des images
- **Recherche d'images par le contenu** : Trouve des images similaires dans la base de données pour aider à la prise de décision
- **Prédictions haute confiance** : Le vote majoritaire entre plus de 5 modèles garantit des résultats fiables
- **Explicabilité** : Affiche les votes individuels des modèles et les scores de confiance

### Qu'est-ce qui rend ce projet unique ?

- **Gabarit prêt pour hackathon** : Structure claire et bien organisée parfaite pour le développement rapide
- **Approche hybride** : Combine des CNN personnalisés avec des extracteurs pré-entraînés et du ML classique
- **Prédictions robustes** : Le vote par ensemble réduit les erreurs des modèles individuels
- **Prêt pour la production** : Inclut un mode démo et des replis gracieux

## ✨ Fonctionnalités

### 🔍 Pipeline de classification
- **CNN de base personnalisé** : Réseau de neurones convolutionnel conçu spécifiquement
- **Extracteurs de caractéristiques pré-entraînés** : ResNet50, VGG16, DenseNet121
- **Classificateurs ML classiques** : SVM, XGBoost, Random Forest, LightGBM
- **Vote par ensemble** : Vote majoritaire entre plusieurs prédictions de modèles
- **Score de confiance** : Confiance en pourcentage avec détail des votes

### 🖼️ Recherche d'images par le contenu (CBIR)
- **Multiples extracteurs de caractéristiques** : Support pour ResNet50, VGG16, DenseNet121
- **Métriques de distance** : Euclidienne, Cosinus, Manhattan, Chebyshev, et plus
- **K plus proches voisins** : Trouve des images visuellement similaires dans la base de données
- **Similarité visuelle** : Affiche les images similaires avec leurs scores de distance

### 📊 Interface web interactive
- Téléversement d'images par glisser-déposer
- Prédiction en temps réel avec retour visuel
- Visualisation des votes des modèles avec graphiques Plotly
- Exploration d'images similaires
- Design responsive optimisé pour l'usage industriel

## 🏗️ Architecture

### Vue d'ensemble du système

```
IMAGE EN ENTRÉE (224×224)
    ↓
┌───────────────────────────────────────────┐
│         EXTRACTION DE CARACTÉRISTIQUES    │
├───────────────────────────────────────────┤
│  ├─→ CNN de base personnalisé            │
│  ├─→ ResNet50 → Vecteur de features      │
│  ├─→ VGG16 → Vecteur de features         │
│  └─→ DenseNet121 → Vecteur de features   │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│         COUCHE DE CLASSIFICATION          │
├───────────────────────────────────────────┤
│  ├─→ Prédiction directe CNN              │
│  ├─→ Classificateurs SVM (par extractor) │
│  ├─→ Classificateurs XGBoost             │
│  └─→ Classificateurs Random Forest       │
└───────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────┐
│         AGRÉGATION PAR ENSEMBLE           │
├───────────────────────────────────────────┤
│  Vote majoritaire (5+ prédictions)       │
│  Confiance = Pourcentage d'accord         │
└───────────────────────────────────────────┘
    ↓
PRÉDICTION FINALE + CONFIANCE + EXPLICATION
```

### Système à double workflow

#### 1. **Workflow de classification**
```python
Image → Extraction features → Classificateurs multiples → Vote → Résultat
```
- Téléverser l'image du composant industriel
- Extraire les caractéristiques avec les backbones CNN
- Appliquer plusieurs classificateurs
- Agréger les prédictions via vote majoritaire
- Afficher les résultats avec scores de confiance

#### 2. **Workflow CBIR**
```python
Image requête → Extraction features → Base signatures → Recherche K-NN → Images similaires
```
- Extraire les caractéristiques de l'image requête
- Comparer avec la base de signatures pré-calculées
- Récupérer les K images les plus similaires
- Afficher avec métriques de distance

## 🛠️ Technologies

| Catégorie | Technologie | Version |
|----------|-----------|---------|
| **Deep Learning** | PyTorch | 2.10 |
| **Deep Learning** | torchvision | 0.25 |
| **Framework Web** | Streamlit | 1.54 |
| **Framework ML** | scikit-learn | Dernière |
| **Boosting** | XGBoost | 3.1 |
| **Boosting** | LightGBM | Dernière |
| **Traitement données** | NumPy | Dernière |
| **Traitement données** | Pandas | Dernière |
| **Vision par ordinateur** | OpenCV | Dernière |
| **Traitement d'images** | Pillow | Dernière |
| **Visualisation** | Plotly | 6.5 |

## 📦 Installation

### Prérequis
- Python 3.8 ou supérieur
- Gestionnaire de paquets pip
- (Optionnel) GPU compatible CUDA pour l'entraînement

### Configuration

1. **Cloner le dépôt**
```bash
git clone <url-du-dépôt>
cd skema-hackathon
```

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

### Dépendances
Le fichier `requirements.txt` inclut tous les paquets nécessaires :
```
streamlit==1.54.0
torch==2.10.0
torchvision==0.25.0
numpy
pandas
pillow
opencv-python
scikit-learn
xgboost==3.1.0
lightgbm
plotly==6.5.0
```

## 🚀 Utilisation

### Lancer l'application web

1. **Démarrer le serveur Streamlit**
```bash
streamlit run streamlit_app.py
```

2. **Accéder à l'application**
- Ouvrir votre navigateur sur `http://localhost:8501`
- L'application fonctionnera en mode démo si les modèles ne sont pas disponibles

3. **Utiliser l'interface**
- Activer le **Mode Démo** dans la barre latérale si aucun modèle entraîné n'existe
- Téléverser une image (JPG, PNG)
- Cliquer sur **"Analyze Image"** pour obtenir :
  - Prédiction finale (Défectueux / OK)
  - Score de confiance global
  - Votes des modèles individuels
  - Visualisation jauge de confiance
  - Graphique de répartition des votes

### Entraîner les modèles

Exécuter le script d'entraînement pour créer tous les modèles :
```bash
python J2/train_and_save_models_jour2.py
```

Cela va :
- Entraîner le CNN de base à partir de zéro
- Extraire les caractéristiques avec les modèles pré-entraînés (ResNet50, VGG16, DenseNet121)
- Entraîner les classificateurs superficiels (SVM, XGBoost, Random Forest)
- Sauvegarder tous les modèles dans le dossier `models/`
- Sauvegarder les scalers de caractéristiques pour la normalisation

### Créer les signatures CBIR

Générer les bases de données de signatures pour la recherche de similarité :
```bash
python misc/create_signatures.py
```

Cela crée des bases de signatures dans `signatures/` pour :
- Caractéristiques ResNet50
- Caractéristiques VGG16
- Caractéristiques DenseNet121

### Utiliser les notebooks Jupyter

Le projet inclut des notebooks pour l'exploration et l'expérimentation :

1. **Notebooks Jour 1** : `01.1hackathon_notebook_template_jour1.ipynb`
   - Exploration des données
   - Prétraitement de base
   - Expériences de modèles initiaux

2. **Notebooks Jour 2** : Situés dans `J2/`
   - Entraînement de modèles avancés
   - Extraction de caractéristiques
   - Évaluation des modèles

3. **Ouvrir avec Jupyter**
```bash
jupyter notebook
```

## 📁 Structure du projet

```
skema-hackathon/
├── streamlit_app.py                       # Application Streamlit principale
├── requirements.txt                       # Dépendances Python
├── README.md                              # Ce fichier
│
├── data/                                  # Répertoire des données
│   ├── train/
│   │   ├── defective/                     # Entraînement : composants défectueux
│   │   └── non_defective/                 # Entraînement : composants normaux
│   └── test/
│       ├── defective/                     # Test : composants défectueux
│       └── non_defective/                 # Test : composants normaux
│
├── models/                                # Fichiers des modèles entraînés
│   ├── baseline_cnn.pth                   # Poids PyTorch du CNN
│   ├── resnet50_svm.pkl                   # Classificateur ResNet50 + SVM
│   ├── resnet50_xgboost.pkl               # ResNet50 + XGBoost
│   ├── resnet50_rf.pkl                    # ResNet50 + Random Forest
│   ├── resnet50_scaler.pkl                # Scaler de features pour ResNet50
│   ├── vgg16_*.pkl                        # Modèles et scaler VGG16
│   └── densenet121_*.pkl                  # Modèles et scaler DenseNet121
│
├── signatures/                            # Bases de données signatures CBIR
│   ├── signatures_resnet50.pkl            # Signatures features ResNet50
│   ├── signatures_vgg16.pkl               # Signatures features VGG16
│   └── signatures_densenet121.pkl         # Signatures features DenseNet121
│
├── J2/                                    # Matériaux de développement Jour 2
│   ├── train_and_save_models_jour2.py     # Script d'entraînement des modèles
│   └── [notebooks et expériences]
│
├── J3/                                    # Matériaux Jour 3
│
├── misc/                                  # Utilitaires et alternatives
│   ├── create_signatures.py               # Constructeur de base CBIR
│   ├── streamlit_app_complete.py          # Version complète de l'app
│   └── utils.py                           # Fonctions utilitaires
│
└── 01.1hackathon_notebook_template_jour1.ipynb  # Notebook Jour 1
```

## 🤖 Modèles

### 1. Architecture CNN de base

**BaselineCNN** - Réseau de neurones convolutionnel personnalisé :

```python
Entrée (224×224×3)
    ↓
Conv2D(32) → ReLU → MaxPool
    ↓
Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool
    ↓
Flatten → FC(512) → Dropout → FC(2)
    ↓
Sortie (2 classes)
```

- **Objectif** : Classification binaire directe
- **Classes** : 0 = Non-défectueux, 1 = Défectueux
- **Fichier** : `models/baseline_cnn.pth`

### 2. Extracteurs de caractéristiques par transfert learning

Backbones CNN pré-entraînés (poids ImageNet) :

| Modèle | Dimensions sortie | Points forts |
|-------|------------|-----------|
| **ResNet50** | 2048 | Connexions résiduelles profondes, excellent pour les motifs complexes |
| **VGG16** | 4096 | Architecture simple, bon pour les caractéristiques de texture |
| **DenseNet121** | 1024 | Connexions denses, réutilisation efficace des features |

### 3. Classificateurs classiques

Appliqués aux caractéristiques extraites :

- **Machine à vecteurs de support (SVM)**
  - Noyaux linéaires et RBF
  - Excellent pour les espaces de caractéristiques haute dimension
  
- **XGBoost**
  - Arbres de décision à gradient boosting
  - Gère bien les relations non-linéaires
  
- **Random Forest**
  - Ensemble d'arbres de décision
  - Robuste au surapprentissage
  
- **LightGBM** (optionnel)
  - Gradient boosting rapide
  - Efficace en mémoire

### Convention de nommage des modèles

```
{extracteur}_{classificateur}.pkl
```

**Exemples :**
- `resnet50_svm.pkl` - SVM entraîné sur features ResNet50
- `vgg16_xgboost.pkl` - XGBoost entraîné sur features VGG16
- `densenet121_rf.pkl` - Random Forest sur features DenseNet121

### Scalers de caractéristiques

Chaque extracteur de caractéristiques a un StandardScaler associé :
- `resnet50_scaler.pkl`
- `vgg16_scaler.pkl`
- `densenet121_scaler.pkl`

Les scalers normalisent les caractéristiques à moyenne nulle et variance unitaire avant classification.

### Vote par ensemble

Le système combine tous les modèles disponibles :
1. Collecter les prédictions de tous les modèles
2. Compter les votes pour chaque classe
3. Sélectionner la classe majoritaire
4. Calculer confiance = (votes majoritaires / votes totaux) × 100%

**Exemple :**
- 5 modèles votent : [Défectueux, Défectueux, OK, Défectueux, Défectueux]
- Résultat : Défectueux avec 80% de confiance

## 📊 Données

### Structure du jeu de données

```
data/
├── train/
│   ├── defective/          # Composants défectueux
│   │   ├── defect_001.jpg
│   │   ├── defect_002.jpg
│   │   └── ...
│   └── non_defective/      # Composants normaux
│       ├── normal_001.jpg
│       ├── normal_002.jpg
│       └── ...
└── test/
    ├── defective/
    └── non_defective/
```

### Exigences pour les images

- **Formats** : JPG, PNG, BMP, GIF
- **Taille d'entrée** : Redimensionnement automatique à 224×224 pixels
- **Canaux** : RGB (3 canaux)
- **Contenu** : Composants industriels, pièces ou assemblages
- **Recommandé** : Images claires, bien éclairées avec arrière-plans cohérents

### Mode démo

L'application inclut un **mode démo** qui fonctionne sans modèles entraînés :
- Simule des prédictions avec confiance aléatoire
- Utile pour tester l'UI/UX
- Activer dans la barre latérale : bouton "Demo Mode"

## 🔧 Configuration

### Personnaliser l'ensemble de modèles

Modifier les extracteurs et classificateurs dans les scripts d'entraînement :

```python
# Dans train_and_save_models_jour2.py
extractors = ['resnet50', 'vgg16', 'densenet121']
classifiers = ['svm', 'xgboost', 'rf']
```

### Paramètres CBIR

Ajuster dans la barre latérale de l'app Streamlit :
- **Extracteur de caractéristiques** : ResNet50, VGG16, DenseNet121
- **Métrique de distance** : 
  - Euclidienne (L2)
  - Similarité Cosinus
  - Manhattan (L1)
  - Chebyshev
- **K Voisins** : 1-20 images similaires

### Paramètres d'entraînement

Paramètres courants dans les notebooks d'entraînement :
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
IMG_SIZE = 224
```

## 📈 Métriques de performance

L'approche par ensemble offre :

- **Robustesse** : Réduit l'impact des erreurs de modèles individuels
- **Estimation de confiance** : La distribution des votes indique la certitude
- **Explicabilité** : Voir quels modèles sont d'accord/en désaccord
- **Flexibilité** : Fonctionne même avec des ensembles de modèles partiels
- **Précision** : Typiquement supérieure aux modèles individuels

**Surveillance :**
- Précision globale
- Précision/rappel par classe
- Matrices de confusion
- Contributions des modèles individuels

## 🤝 Contribuer

Ce projet a été développé pour le Hackathon SKEMA. Les contributions et améliorations sont les bienvenues !

### Comment contribuer

1. Forker le dépôt
2. Créer une branche de fonctionnalité
   ```bash
   git checkout -b feature/amelioration-incroyable
   ```
3. Commiter vos modifications
   ```bash
   git commit -am 'Ajout amélioration incroyable'
   ```
4. Pousser vers la branche
   ```bash
   git push origin feature/amelioration-incroyable
   ```
5. Créer une Pull Request

### Pistes d'amélioration

- [ ] Ajouter plus d'extracteurs pré-entraînés (EfficientNet, Vision Transformer)
- [ ] Implémenter le vote pondéré (au lieu du vote majoritaire simple)
- [ ] Ajouter la calibration des modèles pour de meilleurs scores de confiance
- [ ] Support de la classification multi-classes de défauts
- [ ] Implémenter des pipelines d'augmentation de données
- [ ] Ajouter l'apprentissage actif pour un étiquetage efficace
- [ ] Créer un endpoint API REST
- [ ] Ajouter la surveillance des performances des modèles
- [ ] Implémenter un framework de tests A/B

## 📝 Licence

Ce projet a été créé pour le Hackathon SKEMA. Veuillez consulter les organisateurs pour les informations de licence.

## 🙏 Remerciements

- **SKEMA Business School** pour l'organisation du hackathon
- **L'équipe PyTorch** pour l'excellent framework de deep learning
- **Streamlit** pour le framework web intuitif
- **torchvision** pour les modèles pré-entraînés
- **La communauté ML open-source** pour les outils précieux

## 📧 Support

Pour des questions, problèmes ou suggestions :
- Ouvrir une issue sur GitHub
- Contacter les mainteneurs du projet
- Consulter la [démo en ligne](https://skema-hackathon.streamlit.app)

---

**Développé avec ❤️ pour le Hackathon SKEMA**

**Bonne détection ! 🔍🏭**
