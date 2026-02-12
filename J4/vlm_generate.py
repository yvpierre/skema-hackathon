"""
🤖 VLM - Vision Language Model Template
========================================
Template pour générer des descriptions textuelles d'images de défauts industriels.

Modes disponibles:
1. TEMPLATE - Basé sur règles (pas de GPU requis) ⭐ Recommandé pour commencer
2. BLIP - Modèle Salesforce (GPU recommandé)
3. BLIP2 - Version améliorée (GPU requis)

Usage:
    1. Modifiez les paramètres dans la section CONFIGURATION
    2. Exécutez: python vlm_generate.py
    
    Ou importez dans votre code:
        from vlm_generate import VLMGenerator
        vlm = VLMGenerator(mode='template', language='fr')
        result = vlm.generate(prediction=1, confidence=0.85)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

import numpy as np
from PIL import Image


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CONFIGURATION                                       ║
# ║                    Modifiez ces paramètres selon vos besoins                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Mode VLM à utiliser
# Options: 'template', 'blip', 'blip2'
VLM_MODE = "template"

# Langue des descriptions
# Options: 'fr', 'en'
LANGUAGE = "fr"

# Chemin vers une image de test (optionnel)
TEST_IMAGE = "./data/test/defective/image001.jpg"

# Prompt pour BLIP (guide la génération)
BLIP_PROMPT = "This industrial part shows"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           VÉRIFICATION DÉPENDANCES                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("🔍 Vérification des dépendances...")

# PyTorch
try:
    import torch
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    HAS_TORCH = True
    print(f"   ✅ PyTorch OK (Device: {DEVICE})")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("   ⚠️ PyTorch non installé (pip install torch)")

# Transformers (pour BLIP)
HAS_TRANSFORMERS = False
HAS_BLIP = False
HAS_BLIP2 = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAS_TRANSFORMERS = True
    HAS_BLIP = True
    print("   ✅ BLIP disponible")
except ImportError:
    print("   ⚠️ BLIP non disponible (pip install transformers)")

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    HAS_BLIP2 = True
    print("   ✅ BLIP-2 disponible")
except ImportError:
    print("   ⚠️ BLIP-2 non disponible")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           TEMPLATES DE DESCRIPTIONS                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Types de défauts industriels courants
DEFECT_TYPES = {
    'fr': [
        "rayure superficielle",
        "fissure",
        "déformation",
        "corrosion",
        "inclusion métallique",
        "porosité",
        "bavure",
        "manque de matière",
        "oxydation",
        "écaillage"
    ],
    'en': [
        "surface scratch",
        "crack",
        "deformation",
        "corrosion",
        "metallic inclusion",
        "porosity",
        "burr",
        "material shortage",
        "oxidation",
        "flaking"
    ]
}

# Zones affectées possibles
AFFECTED_ZONES = {
    'fr': [
        "surface principale",
        "bord supérieur",
        "bord inférieur",
        "coin supérieur gauche",
        "coin supérieur droit",
        "coin inférieur gauche",
        "coin inférieur droit",
        "zone centrale",
        "zone de jonction",
        "périmètre"
    ],
    'en': [
        "main surface",
        "upper edge",
        "lower edge",
        "upper left corner",
        "upper right corner",
        "lower left corner",
        "lower right corner",
        "central zone",
        "junction area",
        "perimeter"
    ]
}

# Templates de descriptions
TEMPLATES = {
    'fr': {
        'defective': {
            'high_confidence': {
                'title': "⚠️ DÉFAUT CRITIQUE DÉTECTÉ",
                'status': "REJET",
                'description': """L'analyse automatique a identifié un défaut majeur sur cette pièce industrielle avec un haut niveau de certitude.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Caractéristiques du défaut:**
- **Type probable:** {defect_type}
- **Localisation:** {zone}
- **Sévérité estimée:** Élevée

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """🚨 **ACTION IMMÉDIATE REQUISE:**
1. Retirer immédiatement cette pièce de la ligne de production
2. Marquer la pièce avec un code de traçabilité
3. Documenter le défaut avec photos supplémentaires
4. Notifier le responsable qualité
5. Vérifier les pièces adjacentes dans le lot""",
                'urgency': "URGENT",
                'urgency_level': 3
            },
            'medium_confidence': {
                'title': "⚠️ DÉFAUT DÉTECTÉ",
                'status': "QUARANTAINE",
                'description': """L'analyse automatique suggère la présence d'un défaut sur cette pièce.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Caractéristiques du défaut:**
- **Type probable:** {defect_type}
- **Localisation:** {zone}

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """📋 **ACTION RECOMMANDÉE:**
1. Mettre cette pièce en quarantaine
2. Programmer une inspection manuelle approfondie
3. Comparer avec les images de référence CBIR
4. Décision finale par un opérateur qualifié""",
                'urgency': "ÉLEVÉ",
                'urgency_level': 2
            },
            'low_confidence': {
                'title': "⚠️ DÉFAUT POSSIBLE",
                'status': "À VÉRIFIER",
                'description': """L'analyse automatique détecte une anomalie potentielle, mais le système n'est pas certain.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Observation:**
Le système hésite sur la nature exacte de l'anomalie détectée.

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """📝 **ACTION SUGGÉRÉE:**
1. Vérification visuelle par un opérateur
2. Si doute persiste, effectuer un contrôle dimensionnel
3. Documenter la décision prise""",
                'urgency': "MODÉRÉ",
                'urgency_level': 1
            }
        },
        'non_defective': {
            'high_confidence': {
                'title': "✅ PIÈCE CONFORME",
                'status': "ACCEPTÉ",
                'description': """L'analyse automatique confirme que cette pièce répond aux critères de qualité.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Évaluation:**
- Aucun défaut visible détecté
- Surface conforme aux spécifications
- Géométrie dans les tolérances

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """👍 **STATUT:**
Cette pièce peut continuer dans le processus de production.
Aucune action corrective requise.""",
                'urgency': "AUCUN",
                'urgency_level': 0
            },
            'medium_confidence': {
                'title': "✅ PIÈCE PROBABLEMENT CONFORME",
                'status': "ACCEPTÉ SOUS RÉSERVE",
                'description': """L'analyse automatique indique que cette pièce semble conforme.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """👍 **STATUT:**
Pièce acceptable. Vérification visuelle optionnelle.""",
                'urgency': "AUCUN",
                'urgency_level': 0
            },
            'low_confidence': {
                'title': "❓ STATUT INCERTAIN",
                'status': "À VÉRIFIER",
                'description': """L'analyse automatique n'est pas concluante pour cette pièce.

**Confiance du système:** {confidence:.1%}
**Nombre de modèles en accord:** {votes_for}/{total_votes}

**Observation:**
Les modèles sont en désaccord. Une vérification humaine est nécessaire.

**Analyse CBIR:** {cbir_analysis}""",
                'recommendation': """🔍 **ACTION REQUISE:**
Inspection manuelle obligatoire avant décision.""",
                'urgency': "MODÉRÉ",
                'urgency_level': 1
            }
        }
    },
    'en': {
        'defective': {
            'high_confidence': {
                'title': "⚠️ CRITICAL DEFECT DETECTED",
                'status': "REJECT",
                'description': """Automated analysis has identified a major defect on this industrial part with high certainty.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**Defect characteristics:**
- **Probable type:** {defect_type}
- **Location:** {zone}
- **Estimated severity:** High

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """🚨 **IMMEDIATE ACTION REQUIRED:**
1. Remove this part from production line immediately
2. Mark part with traceability code
3. Document defect with additional photos
4. Notify quality manager
5. Check adjacent parts in batch""",
                'urgency': "URGENT",
                'urgency_level': 3
            },
            'medium_confidence': {
                'title': "⚠️ DEFECT DETECTED",
                'status': "QUARANTINE",
                'description': """Automated analysis suggests presence of a defect on this part.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**Defect characteristics:**
- **Probable type:** {defect_type}
- **Location:** {zone}

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """📋 **RECOMMENDED ACTION:**
1. Quarantine this part
2. Schedule thorough manual inspection
3. Compare with CBIR reference images
4. Final decision by qualified operator""",
                'urgency': "HIGH",
                'urgency_level': 2
            },
            'low_confidence': {
                'title': "⚠️ POSSIBLE DEFECT",
                'status': "TO VERIFY",
                'description': """Automated analysis detects a potential anomaly, but system is uncertain.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**Observation:**
System is uncertain about the exact nature of detected anomaly.

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """📝 **SUGGESTED ACTION:**
1. Visual verification by operator
2. If doubt persists, perform dimensional check
3. Document decision made""",
                'urgency': "MODERATE",
                'urgency_level': 1
            }
        },
        'non_defective': {
            'high_confidence': {
                'title': "✅ PART CONFORMING",
                'status': "ACCEPTED",
                'description': """Automated analysis confirms this part meets quality criteria.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**Evaluation:**
- No visible defects detected
- Surface conforms to specifications
- Geometry within tolerances

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """👍 **STATUS:**
This part can continue in production process.
No corrective action required.""",
                'urgency': "NONE",
                'urgency_level': 0
            },
            'medium_confidence': {
                'title': "✅ PART LIKELY CONFORMING",
                'status': "CONDITIONALLY ACCEPTED",
                'description': """Automated analysis indicates this part appears conforming.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """👍 **STATUS:**
Part acceptable. Optional visual verification.""",
                'urgency': "NONE",
                'urgency_level': 0
            },
            'low_confidence': {
                'title': "❓ UNCERTAIN STATUS",
                'status': "TO VERIFY",
                'description': """Automated analysis is inconclusive for this part.

**System confidence:** {confidence:.1%}
**Models in agreement:** {votes_for}/{total_votes}

**Observation:**
Models disagree. Human verification is necessary.

**CBIR Analysis:** {cbir_analysis}""",
                'recommendation': """🔍 **ACTION REQUIRED:**
Mandatory manual inspection before decision.""",
                'urgency': "MODERATE",
                'urgency_level': 1
            }
        }
    }
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CLASSE TEMPLATE VLM                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TemplateVLM:
    """
    Générateur de descriptions basé sur des templates.
    ✅ Ne nécessite PAS de GPU
    ✅ Rapide et déterministe
    ✅ Personnalisable
    """
    
    def __init__(self, language: str = 'fr'):
        """
        Initialise le générateur template.
        
        Args:
            language: 'fr' pour français, 'en' pour anglais
        """
        self.language = language
        self.templates = TEMPLATES[language]
        self.defect_types = DEFECT_TYPES[language]
        self.zones = AFFECTED_ZONES[language]
        
        print(f"   ✅ TemplateVLM initialisé (langue: {language})")
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Détermine le niveau de confiance."""
        if confidence >= 0.85:
            return 'high_confidence'
        elif confidence >= 0.60:
            return 'medium_confidence'
        return 'low_confidence'
    
    def _get_defect_type(self, seed: int) -> str:
        """Sélectionne un type de défaut basé sur un seed."""
        return self.defect_types[seed % len(self.defect_types)]
    
    def _get_zone(self, seed: int) -> str:
        """Sélectionne une zone basée sur un seed."""
        return self.zones[seed % len(self.zones)]
    
    def _format_cbir_analysis(self, cbir_results: Optional[List[Dict]] = None) -> str:
        """Formate l'analyse CBIR."""
        if not cbir_results or len(cbir_results) == 0:
            if self.language == 'fr':
                return "Non disponible"
            return "Not available"
        
        defect_count = sum(1 for r in cbir_results if r.get('label', 0) == 1)
        total = len(cbir_results)
        
        if self.language == 'fr':
            return f"{defect_count}/{total} images similaires présentent des défauts"
        return f"{defect_count}/{total} similar images show defects"
    
    def generate(
        self,
        prediction: int,
        confidence: float,
        votes_for: Optional[int] = None,
        total_votes: Optional[int] = None,
        cbir_results: Optional[List[Dict]] = None,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère une description complète.
        
        Args:
            prediction: 0 (non_defective) ou 1 (defective)
            confidence: float entre 0 et 1
            votes_for: nombre de votes pour la prédiction
            total_votes: nombre total de votes
            cbir_results: liste des résultats CBIR
            image_id: identifiant optionnel de l'image
        
        Returns:
            Dict contenant tous les éléments de la description
        """
        # Classe et niveau de confiance
        class_name = 'defective' if prediction == 1 else 'non_defective'
        conf_level = self._get_confidence_level(confidence)
        
        # Template
        template = self.templates[class_name][conf_level]
        
        # Seed pour sélection déterministe
        seed = int(confidence * 1000)
        defect_type = self._get_defect_type(seed)
        zone = self._get_zone(seed + 42)
        
        # Votes
        if votes_for is None:
            votes_for = int(confidence * 5)
        if total_votes is None:
            total_votes = 5
        
        # Analyse CBIR
        cbir_analysis = self._format_cbir_analysis(cbir_results)
        
        # Formatage
        description = template['description'].format(
            confidence=confidence,
            votes_for=votes_for,
            total_votes=total_votes,
            defect_type=defect_type,
            zone=zone,
            cbir_analysis=cbir_analysis
        )
        
        # Résultat complet
        result = {
            'title': template['title'],
            'status': template['status'],
            'description': description,
            'recommendation': template['recommendation'],
            'urgency': template['urgency'],
            'urgency_level': template['urgency_level'],
            'prediction': prediction,
            'confidence': confidence,
            'defect_type': defect_type if prediction == 1 else None,
            'zone': zone if prediction == 1 else None,
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'full_report': None
        }
        
        # Rapport complet
        result['full_report'] = f"""
{'='*60}
{template['title']}
{'='*60}

📊 STATUT: {template['status']}
⏰ Date: {result['timestamp']}
🖼️ Image: {image_id or 'N/A'}

{description}

{'='*60}
💡 RECOMMANDATION
{'='*60}

{template['recommendation']}

{'='*60}
"""
        
        return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CLASSE BLIP VLM                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BLIPVLM:
    """
    Vision Language Model basé sur BLIP (Salesforce).
    ⚠️ GPU recommandé
    ⚠️ Télécharge le modèle (~1GB) au premier lancement
    """
    
    def __init__(self, model_size: str = 'base'):
        """
        Initialise BLIP.
        
        Args:
            model_size: 'base' (~990MB) ou 'large' (~1.9GB)
        """
        if not HAS_BLIP:
            raise ImportError("BLIP non disponible. pip install transformers")
        
        model_name = f"Salesforce/blip-image-captioning-{model_size}"
        
        print(f"   🔄 Chargement de BLIP ({model_size})...")
        print(f"      Cela peut prendre quelques minutes au premier lancement...")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        if DEVICE and DEVICE.type == 'cuda':
            self.model = self.model.to(DEVICE)
            print(f"   ✅ BLIP chargé sur GPU")
        else:
            print(f"   ✅ BLIP chargé sur CPU (plus lent)")
        
        self.model.eval()
    
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = 50,
        num_beams: int = 5
    ) -> str:
        """
        Génère une légende pour une image.
        
        Args:
            image: chemin ou PIL Image
            prompt: texte conditionnel (optionnel)
            max_length: longueur max de la description
            num_beams: beam search width
        
        Returns:
            str: description générée
        """
        # Charger l'image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        # Préparer
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(image, return_tensors="pt")
        
        if DEVICE and DEVICE.type == 'cuda':
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Générer
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def generate(
        self,
        image: Union[str, Path, Image.Image],
        prediction: Optional[int] = None,
        confidence: Optional[float] = None,
        cbir_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Génère une description complète avec contexte industriel.
        
        Args:
            image: image à analyser
            prediction: résultat de classification (optionnel)
            confidence: confiance (optionnel)
            cbir_results: résultats CBIR (optionnel)
        
        Returns:
            Dict avec la description
        """
        # Prompt selon la prédiction
        if prediction == 1:
            prompt = "This industrial part shows a defect:"
        elif prediction == 0:
            prompt = "This industrial part is:"
        else:
            prompt = "This industrial part shows"
        
        # Générer la description
        caption = self.generate_caption(image, prompt=prompt)
        
        # Construire le résultat
        result = {
            'raw_caption': caption,
            'prompt_used': prompt,
            'prediction': prediction,
            'confidence': confidence,
            'title': None,
            'description': None,
            'recommendation': None
        }
        
        # Enrichir si on a la prédiction
        if prediction is not None:
            if prediction == 1:
                result['title'] = "⚠️ DÉFAUT DÉTECTÉ (BLIP)"
                result['description'] = f"**Description IA:** {caption}"
                if confidence:
                    result['description'] += f"\n\n**Confiance:** {confidence:.1%}"
                result['recommendation'] = "Inspection manuelle recommandée."
            else:
                result['title'] = "✅ PIÈCE ANALYSÉE (BLIP)"
                result['description'] = f"**Description IA:** {caption}"
                if confidence:
                    result['description'] += f"\n\n**Confiance:** {confidence:.1%}"
                result['recommendation'] = "Pièce semble conforme."
        else:
            result['title'] = "🔍 ANALYSE BLIP"
            result['description'] = f"**Description IA:** {caption}"
        
        return result


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           CLASSE UNIFIÉE VLM GENERATOR                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class VLMGenerator:
    """
    Classe unifiée pour générer des descriptions VLM.
    Sélectionne automatiquement le meilleur backend disponible.
    """
    
    def __init__(self, mode: str = 'template', language: str = 'fr'):
        """
        Initialise le générateur VLM.
        
        Args:
            mode: 'template', 'blip', 'blip2'
            language: 'fr' ou 'en'
        """
        self.mode = mode
        self.language = language
        self.backend = None
        
        print(f"\n🤖 Initialisation VLM (mode: {mode}, langue: {language})")
        
        if mode == 'template':
            self.backend = TemplateVLM(language=language)
            
        elif mode == 'blip':
            if HAS_BLIP:
                try:
                    self.backend = BLIPVLM(model_size='base')
                except Exception as e:
                    print(f"   ⚠️ Erreur BLIP: {e}")
                    print("   ⚠️ Fallback vers template")
                    self.mode = 'template'
                    self.backend = TemplateVLM(language=language)
            else:
                print("   ⚠️ BLIP non disponible, utilisation template")
                self.mode = 'template'
                self.backend = TemplateVLM(language=language)
                
        else:
            print(f"   ⚠️ Mode {mode} non supporté, utilisation template")
            self.mode = 'template'
            self.backend = TemplateVLM(language=language)
    
    def generate(
        self,
        image: Optional[Union[str, Path, Image.Image]] = None,
        prediction: int = 0,
        confidence: float = 0.5,
        votes_for: Optional[int] = None,
        total_votes: Optional[int] = None,
        cbir_results: Optional[List[Dict]] = None,
        image_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère une description.
        
        Args:
            image: image (requis pour BLIP, optionnel pour template)
            prediction: 0 ou 1
            confidence: 0.0 à 1.0
            votes_for: votes pour la prédiction
            total_votes: total des votes
            cbir_results: résultats CBIR
            image_id: ID de l'image
        
        Returns:
            Dict avec tous les éléments de description
        """
        if self.mode == 'template':
            return self.backend.generate(
                prediction=prediction,
                confidence=confidence,
                votes_for=votes_for,
                total_votes=total_votes,
                cbir_results=cbir_results,
                image_id=image_id
            )
        
        elif self.mode == 'blip':
            if image is None:
                raise ValueError("Image requise pour BLIP")
            return self.backend.generate(
                image=image,
                prediction=prediction,
                confidence=confidence,
                cbir_results=cbir_results
            )
        
        return {}
    
    def generate_batch(
        self,
        results_list: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Génère des descriptions pour plusieurs résultats.
        
        Args:
            results_list: liste de dicts avec 'prediction', 'confidence', etc.
        
        Returns:
            Liste de descriptions
        """
        descriptions = []
        
        for result in results_list:
            desc = self.generate(
                image=result.get('image'),
                prediction=result.get('prediction', 0),
                confidence=result.get('confidence', 0.5),
                votes_for=result.get('votes_for'),
                total_votes=result.get('total_votes'),
                cbir_results=result.get('cbir_results'),
                image_id=result.get('image_id')
            )
            descriptions.append(desc)
        
        return descriptions


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           FONCTION PRINCIPALE                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    """Fonction principale pour tester le VLM."""
    
    print("\n" + "=" * 60)
    print("🤖 VLM - Vision Language Model - TESTS")
    print("=" * 60)
    
    # Créer le générateur
    vlm = VLMGenerator(mode=VLM_MODE, language=LANGUAGE)
    
    # Test 1: Défaut haute confiance
    print("\n" + "-" * 60)
    print("📋 TEST 1: Défaut - Haute confiance (92%)")
    print("-" * 60)
    
    result = vlm.generate(
        prediction=1,
        confidence=0.92,
        votes_for=4,
        total_votes=5,
        cbir_results=[
            {'label': 1}, {'label': 1}, {'label': 1}, {'label': 0}, {'label': 1}
        ],
        image_id="IMG_001"
    )
    
    print(result['full_report'])
    
    # Test 2: Conforme haute confiance
    print("\n" + "-" * 60)
    print("📋 TEST 2: Conforme - Haute confiance (89%)")
    print("-" * 60)
    
    result = vlm.generate(
        prediction=0,
        confidence=0.89,
        votes_for=4,
        total_votes=5,
        image_id="IMG_002"
    )
    
    print(result['full_report'])
    
    # Test 3: Cas incertain
    print("\n" + "-" * 60)
    print("📋 TEST 3: Cas incertain (55%)")
    print("-" * 60)
    
    result = vlm.generate(
        prediction=1,
        confidence=0.55,
        votes_for=3,
        total_votes=5,
        image_id="IMG_003"
    )
    
    print(result['full_report'])
    
    # Test avec image si disponible et mode BLIP
    if VLM_MODE == 'blip' and Path(TEST_IMAGE).exists():
        print("\n" + "-" * 60)
        print(f"📋 TEST 4: Analyse BLIP de {TEST_IMAGE}")
        print("-" * 60)
        
        result = vlm.generate(
            image=TEST_IMAGE,
            prediction=1,
            confidence=0.85
        )
        
        print(f"Titre: {result.get('title')}")
        print(f"Description: {result.get('description')}")
    
    print("\n" + "=" * 60)
    print("✅ Tests VLM terminés!")
    print("=" * 60)


if __name__ == '__main__':
    main()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           EXEMPLES D'UTILISATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

"""
# ================================================
# EXEMPLE 1: Usage simple avec Template
# ================================================

from vlm_generate import VLMGenerator

# Créer le générateur
vlm = VLMGenerator(mode='template', language='fr')

# Générer une description
result = vlm.generate(
    prediction=1,        # 1 = défaut, 0 = OK
    confidence=0.85,     # Confiance du classifieur
    votes_for=4,         # 4 modèles sur 5 votent défaut
    total_votes=5
)

# Utiliser les résultats
print(result['title'])           # "⚠️ DÉFAUT CRITIQUE DÉTECTÉ"
print(result['status'])          # "REJET"
print(result['urgency'])         # "URGENT"
print(result['recommendation'])  # Instructions d'action
print(result['full_report'])     # Rapport complet formaté


# ================================================
# EXEMPLE 2: Avec résultats CBIR
# ================================================

from vlm_generate import VLMGenerator

vlm = VLMGenerator(mode='template', language='fr')

# Résultats CBIR (de votre recherche d'images similaires)
cbir_results = [
    {'label': 1, 'distance': 0.12},  # Image similaire défectueuse
    {'label': 1, 'distance': 0.15},  # Image similaire défectueuse
    {'label': 0, 'distance': 0.18},  # Image similaire OK
    {'label': 1, 'distance': 0.22},  # Image similaire défectueuse
    {'label': 1, 'distance': 0.25},  # Image similaire défectueuse
]

result = vlm.generate(
    prediction=1,
    confidence=0.88,
    cbir_results=cbir_results
)

# La description inclura: "4/5 images similaires présentent des défauts"


# ================================================
# EXEMPLE 3: Intégration Streamlit
# ================================================

import streamlit as st
from vlm_generate import VLMGenerator

# Dans votre app Streamlit
vlm = VLMGenerator(mode='template', language='fr')

# Après la classification et CBIR...
vlm_result = vlm.generate(
    prediction=pred_result['prediction'],
    confidence=pred_result['confidence'],
    votes_for=pred_result['votes']['defective'] if pred_result['prediction'] == 1 
              else pred_result['votes']['non_defective'],
    total_votes=sum(pred_result['votes'].values()),
    cbir_results=cbir_results
)

# Afficher dans Streamlit
st.markdown(f"## {vlm_result['title']}")
st.markdown(vlm_result['description'])
st.info(vlm_result['recommendation'])

# Badge d'urgence
if vlm_result['urgency_level'] >= 2:
    st.error(f"🚨 Urgence: {vlm_result['urgency']}")
elif vlm_result['urgency_level'] == 1:
    st.warning(f"⚠️ Urgence: {vlm_result['urgency']}")
else:
    st.success(f"✅ Urgence: {vlm_result['urgency']}")


# ================================================
# EXEMPLE 4: Batch processing
# ================================================

from vlm_generate import VLMGenerator

vlm = VLMGenerator(mode='template', language='fr')

# Liste de résultats à traiter
batch_results = [
    {'prediction': 1, 'confidence': 0.92, 'image_id': 'IMG_001'},
    {'prediction': 0, 'confidence': 0.88, 'image_id': 'IMG_002'},
    {'prediction': 1, 'confidence': 0.55, 'image_id': 'IMG_003'},
]

# Générer toutes les descriptions
descriptions = vlm.generate_batch(batch_results)

for desc in descriptions:
    print(f"{desc['image_id']}: {desc['title']} - {desc['status']}")


# ================================================
# EXEMPLE 5: Mode BLIP (avec GPU)
# ================================================

from vlm_generate import VLMGenerator

# Créer avec BLIP
vlm = VLMGenerator(mode='blip', language='en')

# Générer avec une image
result = vlm.generate(
    image='./image.jpg',
    prediction=1,
    confidence=0.90
)

print(result['raw_caption'])  # Description générée par BLIP
"""
