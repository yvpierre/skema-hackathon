# 🤖 VLM - Vision Language Model

**Template pour générer des descriptions textuelles d'images de défauts industriels**

---

## 📖 Table des Matières

- [Introduction](#-introduction)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Modes Disponibles](#-modes-disponibles)
- [Usage Détaillé](#-usage-détaillé)
- [Intégration Streamlit](#-intégration-streamlit)
- [Structure des Résultats](#-structure-des-résultats)
- [Personnalisation](#-personnalisation)
- [Exemples](#-exemples)

---

## 🎯 Introduction

Ce module génère des **descriptions textuelles automatiques** pour les images analysées par le système de détection de défauts. Il transforme les résultats numériques (prédiction, confiance, CBIR) en **rapports compréhensibles** pour les opérateurs.

### Pourquoi utiliser un VLM?

| Sans VLM | Avec VLM |
|----------|----------|
| `prediction: 1, confidence: 0.87` | ⚠️ **DÉFAUT DÉTECTÉ** - L'analyse révèle une anomalie majeure. Type probable: fissure. Action immédiate requise. |
| Données brutes | Rapport actionnable |
| Pour développeurs | Pour opérateurs |

---

## 📦 Installation

### Dépendances minimales (Mode Template)

```bash
pip install numpy pillow
```

### Dépendances complètes (Mode BLIP)

```bash
pip install torch torchvision transformers pillow numpy
```

---

## 🚀 Quick Start

### Étape 1: Modifier la configuration

Ouvrez `vlm_generate.py` et modifiez:

```python
# ╔══════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════════════════╝

VLM_MODE = "template"    # 'template' ou 'blip'
LANGUAGE = "fr"          # 'fr' ou 'en'
```

### Étape 2: Utiliser dans votre code

```python
from vlm_generate import VLMGenerator

# Créer le générateur
vlm = VLMGenerator(mode='template', language='fr')

# Générer une description
result = vlm.generate(
    prediction=1,        # 1 = défaut, 0 = OK
    confidence=0.85,     # Confiance (0-1)
    votes_for=4,         # Votes pour cette prédiction
    total_votes=5        # Total des votes
)

# Afficher
print(result['title'])          # ⚠️ DÉFAUT CRITIQUE DÉTECTÉ
print(result['status'])         # REJET
print(result['recommendation']) # Instructions d'action
print(result['full_report'])    # Rapport complet
```

### Étape 3: Tester

```bash
python vlm_generate.py
```

---

## 🔧 Modes Disponibles

### 1. Mode TEMPLATE ⭐ Recommandé

```python
vlm = VLMGenerator(mode='template', language='fr')
```

| Avantages | Inconvénients |
|-----------|---------------|
| ✅ Pas de GPU requis | ❌ Descriptions basées sur règles |
| ✅ Rapide et déterministe | ❌ Pas d'analyse visuelle réelle |
| ✅ Personnalisable facilement | |
| ✅ Fonctionne hors-ligne | |

### 2. Mode BLIP

```python
vlm = VLMGenerator(mode='blip', language='en')
```

| Avantages | Inconvénients |
|-----------|---------------|
| ✅ Analyse visuelle réelle | ❌ GPU recommandé |
| ✅ Descriptions naturelles | ❌ Téléchargement ~1GB |
| ✅ Adaptatif au contenu | ❌ Plus lent |

---

## 📝 Usage Détaillé

### Paramètres de `generate()`

| Paramètre | Type | Requis | Description |
|-----------|------|--------|-------------|
| `prediction` | int | ✅ | 0 (OK) ou 1 (défaut) |
| `confidence` | float | ✅ | Confiance entre 0.0 et 1.0 |
| `votes_for` | int | ❌ | Nombre de votes pour la prédiction |
| `total_votes` | int | ❌ | Nombre total de modèles |
| `cbir_results` | list | ❌ | Résultats CBIR |
| `image_id` | str | ❌ | Identifiant de l'image |
| `image` | PIL/path | ❌* | Image (requis pour BLIP) |

### Avec résultats CBIR

```python
# Résultats de votre recherche CBIR
cbir_results = [
    {'label': 1, 'distance': 0.12},
    {'label': 1, 'distance': 0.15},
    {'label': 0, 'distance': 0.18},
    {'label': 1, 'distance': 0.22},
    {'label': 1, 'distance': 0.25},
]

result = vlm.generate(
    prediction=1,
    confidence=0.88,
    cbir_results=cbir_results
)

# La description inclura:
# "Analyse CBIR: 4/5 images similaires présentent des défauts"
```

### Traitement par lots

```python
batch_results = [
    {'prediction': 1, 'confidence': 0.92, 'image_id': 'IMG_001'},
    {'prediction': 0, 'confidence': 0.88, 'image_id': 'IMG_002'},
    {'prediction': 1, 'confidence': 0.55, 'image_id': 'IMG_003'},
]

descriptions = vlm.generate_batch(batch_results)

for desc in descriptions:
    print(f"{desc['image_id']}: {desc['status']}")
```

---

## 🖥️ Intégration Streamlit

### Code complet

```python
import streamlit as st
from vlm_generate import VLMGenerator

# Initialiser le VLM (une seule fois)
@st.cache_resource
def load_vlm():
    return VLMGenerator(mode='template', language='fr')

vlm = load_vlm()

# Après avoir obtenu les résultats de classification et CBIR...
vlm_result = vlm.generate(
    prediction=pred_result['prediction'],
    confidence=pred_result['confidence'],
    votes_for=pred_result['votes']['defective'] if pred_result['prediction'] == 1 
              else pred_result['votes']['non_defective'],
    total_votes=sum(pred_result['votes'].values()),
    cbir_results=cbir_results
)

# Afficher le titre
st.markdown(f"## {vlm_result['title']}")

# Afficher le statut avec couleur
if vlm_result['prediction'] == 1:
    st.error(f"**Statut:** {vlm_result['status']}")
else:
    st.success(f"**Statut:** {vlm_result['status']}")

# Description
st.markdown(vlm_result['description'])

# Recommandation
st.info(vlm_result['recommendation'])

# Badge d'urgence
urgency_colors = {
    0: ("✅", "success"),
    1: ("⚠️", "warning"),
    2: ("🔶", "warning"),
    3: ("🚨", "error")
}
icon, method = urgency_colors.get(vlm_result['urgency_level'], ("❓", "info"))
getattr(st, method)(f"{icon} Urgence: {vlm_result['urgency']}")
```

---

## 📊 Structure des Résultats

Le dictionnaire retourné par `generate()` contient:

```python
{
    # Éléments principaux
    'title': str,           # "⚠️ DÉFAUT CRITIQUE DÉTECTÉ"
    'status': str,          # "REJET", "QUARANTAINE", "ACCEPTÉ", etc.
    'description': str,     # Description complète formatée
    'recommendation': str,  # Instructions d'action
    
    # Urgence
    'urgency': str,         # "URGENT", "ÉLEVÉ", "MODÉRÉ", "AUCUN"
    'urgency_level': int,   # 0-3 (0=aucun, 3=urgent)
    
    # Métadonnées
    'prediction': int,      # 0 ou 1
    'confidence': float,    # 0.0-1.0
    'defect_type': str,     # Type de défaut (si défaut)
    'zone': str,            # Zone affectée (si défaut)
    'timestamp': str,       # ISO datetime
    'image_id': str,        # ID de l'image
    
    # Rapport complet
    'full_report': str      # Rapport formaté complet
}
```

### Niveaux d'urgence

| Niveau | Valeur | Signification |
|--------|--------|---------------|
| `URGENT` | 3 | Défaut haute confiance - Action immédiate |
| `ÉLEVÉ` | 2 | Défaut moyenne confiance - Quarantaine |
| `MODÉRÉ` | 1 | Incertain - Vérification requise |
| `AUCUN` | 0 | Conforme - Continuer production |

---

## 🎨 Personnalisation

### Ajouter des types de défauts

Dans `vlm_generate.py`, modifiez `DEFECT_TYPES`:

```python
DEFECT_TYPES = {
    'fr': [
        "rayure superficielle",
        "fissure",
        "déformation",
        # Ajoutez vos types ici
        "soudure défectueuse",
        "contamination",
    ],
    'en': [
        "surface scratch",
        "crack",
        # ...
    ]
}
```

### Modifier les templates

Modifiez la section `TEMPLATES` pour personnaliser:

```python
TEMPLATES = {
    'fr': {
        'defective': {
            'high_confidence': {
                'title': "⚠️ VOTRE TITRE PERSONNALISÉ",
                'status': "REJET",
                'description': """Votre description personnalisée...
                
**Confiance:** {confidence:.1%}
**Type:** {defect_type}
""",
                'recommendation': "Vos instructions...",
                'urgency': "CRITIQUE",
                'urgency_level': 3
            },
            # ...
        }
    }
}
```

---

## 💡 Exemples

### Exemple 1: Défaut critique

```python
result = vlm.generate(prediction=1, confidence=0.95, votes_for=5, total_votes=5)
```

**Sortie:**
```
============================================================
⚠️ DÉFAUT CRITIQUE DÉTECTÉ
============================================================

📊 STATUT: REJET

L'analyse automatique a identifié un défaut majeur sur cette pièce 
industrielle avec un haut niveau de certitude.

Confiance du système: 95.0%
Nombre de modèles en accord: 5/5

Caractéristiques du défaut:
- Type probable: fissure
- Localisation: surface principale
- Sévérité estimée: Élevée

============================================================
💡 RECOMMANDATION
============================================================

🚨 ACTION IMMÉDIATE REQUISE:
1. Retirer immédiatement cette pièce de la ligne de production
2. Marquer la pièce avec un code de traçabilité
3. Documenter le défaut avec photos supplémentaires
4. Notifier le responsable qualité
5. Vérifier les pièces adjacentes dans le lot
```

### Exemple 2: Pièce conforme

```python
result = vlm.generate(prediction=0, confidence=0.92, votes_for=5, total_votes=5)
```

**Sortie:**
```
============================================================
✅ PIÈCE CONFORME
============================================================

📊 STATUT: ACCEPTÉ

L'analyse automatique confirme que cette pièce répond aux critères 
de qualité.

Confiance du système: 92.0%

Évaluation:
- Aucun défaut visible détecté
- Surface conforme aux spécifications
- Géométrie dans les tolérances

============================================================
💡 RECOMMANDATION
============================================================

👍 STATUT:
Cette pièce peut continuer dans le processus de production.
Aucune action corrective requise.
```

### Exemple 3: Cas incertain

```python
result = vlm.generate(prediction=1, confidence=0.55, votes_for=3, total_votes=5)
```

**Sortie:**
```
============================================================
⚠️ DÉFAUT POSSIBLE
============================================================

📊 STATUT: À VÉRIFIER

L'analyse automatique détecte une anomalie potentielle, mais le 
système n'est pas certain.

Confiance du système: 55.0%
Nombre de modèles en accord: 3/5

Observation:
Le système hésite sur la nature exacte de l'anomalie détectée.

============================================================
💡 RECOMMANDATION
============================================================

📝 ACTION SUGGÉRÉE:
1. Vérification visuelle par un opérateur
2. Si doute persiste, effectuer un contrôle dimensionnel
3. Documenter la décision prise
```

---

## 🐛 Troubleshooting

| Problème | Solution |
|----------|----------|
| `ImportError: transformers` | Mode BLIP requis: `pip install transformers` |
| Description en anglais | Vérifier `LANGUAGE = "fr"` |
| BLIP très lent | Normal sur CPU, utiliser GPU ou mode template |
| Descriptions identiques | Normal pour template (déterministe basé sur confidence) |

---

## 📁 Fichiers

| Fichier | Description |
|---------|-------------|
| `vlm_generate.py` | Script principal VLM |
| `README_VLM.md` | Ce guide |

---

**Bon Hackathon! 🚀**
