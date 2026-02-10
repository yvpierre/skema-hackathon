🏭 Détection de Défauts Industriels – Projet Hackathon
=====================================================

Ce dépôt propose une petite pipeline **de bout en bout** pour la **détection de défauts industriels à partir d’images**, développée pour un hackathon. Il combine :

- Des notebooks Jupyter pour **l’exploration, l’entraînement des modèles et la sauvegarde**.
- Une **application Streamlit** pour téléverser des images et effectuer une **classification défaut / non‑défaut** de manière interactive.
- La prise en charge à la fois d’un **CNN de base personnalisé** et de **réseaux CNN pré‑entraînés comme extracteurs de caractéristiques** (ResNet, VGG, DenseNet, etc.) couplés à des modèles classiques (SVM, Random Forest, XGBoost…).

---
👉 Tester l’application Streamlit ici : [Streamlit app](https://skema-hackathon.streamlit.app)

Objectif du projet
------------------

L’objectif principal est de fournir un **gabarit clair et adapté à un hackathon** pour :

- Concevoir et expérimenter des **modèles de détection de défauts sur images**.
- Mettre rapidement ces modèles à disposition via une **interface Streamlit conviviale**.
- Illustrer comment combiner **extracteurs de caractéristiques profonds** et **modèles de machine learning classiques** au sein d’un ensemble pour des prédictions plus robustes.


Structure du projet
-------------------

- `data/`
	- `train/defective`, `train/non_defective` : images d’entraînement.
	- `test/defective`, `test/non_defective` : images de test.
- `models/`
	- `baseline_cnn.pth` : poids d’exemple du CNN de base (si disponibles).
	- `*.pkl`, `*_scaler.pkl` : modèles classiques (SVM, Random Forest, XGBoost, …) et leurs scalers associés.
- `streamlit_app.py`
	- Application Streamlit principale pour la **classification d’images** avec vote d’ensemble.
	- Gère le téléversement d’images, le pré‑traitement, la prédiction et les visualisations.
- `01.1hackathon_notebook_template_jour1.ipynb`, `01.1hackathon_notebook_template_jour2.ipynb`
	- Notebooks hackathon pour **le jour 1**.
- `J2/`
	- Notebooks et scripts **du jour 2**.
- `misc/`
	- Utilitaires alternatifs / expérimentaux (par ex. `streamlit_app_bis.py`, `utils.py`).

---

Composants principaux
---------------------

### 1. CNN de base (BaselineCNN)

Défini dans `streamlit_app.py` sous le nom `BaselineCNN`, il s’agit d’un réseau de neurones convolutionnel simple, entraîné pour classer les images en deux classes :

- **Classe 0** : Non‑défectueux
- **Classe 1** : Défectueux

Les poids correspondants peuvent être enregistrés dans `models/baseline_cnn.pth` et sont chargés automatiquement par l’application s’ils existent.

### 2. Extracteurs de caractéristiques + modèles classiques

L’application utilise également un `FeatureExtractor` qui encapsule des **backbones CNN pré‑entraînés** de `torchvision` (par ex. ResNet50, VGG16, DenseNet121). Ils servent à extraire des vecteurs de caractéristiques fixes, ensuite transmis à des **classifieurs classiques** tels que :

- SVM
- Random Forest
- XGBoost

Ces modèles, ainsi que leurs scalers, sont stockés sous forme de fichiers `*.pkl` et `*_scaler.pkl` dans le dossier `models/`.

### 3. Prédiction par ensemble (ensemble learning)

L’application Streamlit agrège les prédictions de :

- Le CNN de base.
- Tous les modèles classiques disponibles.

Elle applique ensuite un **vote majoritaire** pour produire une décision finale (Défectueux / Non‑défectueux), accompagnée d’un niveau de confiance global et de détails par modèle.

Si aucun modèle entraîné n’est trouvé, l’application peut basculer en **mode démo**, avec des modèles factices, afin que l’interface reste utilisable pendant le développement.

---

Lancer l’application Streamlit
------------------------------

1. **Installer les dépendances**

	```bash
	pip install -r requirements.txt
	```

2. **(Optionnel) Ajouter des modèles entraînés**

	- Placer `baseline_cnn.pth` et tout fichier `*.pkl` / `*_scaler.pkl` dans le dossier `models/`.

3. **Lancer l’application**

	Depuis la racine du projet :

	```bash
	streamlit run streamlit_app.py
	```

4. **Utiliser l’interface web**

	- Ouvrir l’URL affichée par Streamlit (en général `http://localhost:8501`).
	- Dans la barre latérale, activer **Demo Mode** pour utiliser des prédictions simulées si aucun modèle réel n’est disponible.
	- Téléverser une image d’une pièce industrielle.
	- Cliquer sur **« Analyze Image »** pour obtenir :
		- La décision finale (Défectueux / Non‑défectueux) avec une confiance globale.
		- Les votes et niveaux de confiance par modèle.
		- Des visualisations comme un indicateur de confiance (jauge) et la répartition des votes.

---

Utiliser les notebooks
----------------------

Les notebooks (`01.1hackathon_notebook_template_jour1.ipynb`, `01.1hackathon_notebook_template_jour2.ipynb` et ceux dans `J2/`) sont organisés pour vous guider à travers :

- L’exploration des données et le pré‑traitement de base.
- L’entraînement et l’évaluation de modèles de base et plus avancés.
- La sauvegarde des poids et des classifieurs dans le dossier `models/` pour une utilisation ultérieure dans l’application Streamlit.

Vous pouvez les ouvrir dans Jupyter, VS Code ou tout autre environnement compatible et exécuter les cellules pas à pas.

---


