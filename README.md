# skema-hackathon
- Project for Skema's 2026 Hackathon


🏭 Industrial Defect Detection – Hackathon Project
=================================================

This project is a small end‑to‑end pipeline for **industrial defect detection from images**, built for a hackathon. It combines:

- Jupyter notebooks for **exploration, model training and saving**.
- A **Streamlit web app** for interactive image upload and **defect / non‑defect classification**.
- Support for both a custom **CNN baseline** and **pre‑trained CNN feature extractors** (ResNet, VGG, DenseNet, etc.) feeding shallow classifiers.

---

Check the [Streamlit app](https://skema-hackathon.streamlit.app) out

Project Structure
-----------------

- `data/`
	- `train/defective`, `train/non_defective`: training images.
	- `test/defective`, `test/non_defective`: test images.
- `models/`
	- `baseline_cnn.pth`: example trained baseline CNN weights (if available).
	- `*.pkl`, `*_scaler.pkl`: optional shallow models (SVM, RF, XGBoost, …) and associated scalers.
- `streamlit_app.py`
	- Main Streamlit app for **image classification** with ensemble voting.
	- Handles image upload, preprocessing, prediction and visualizations.
- `01.1hackathon_notebook_template_jour1.ipynb`, `01.1hackathon_notebook_template_jour2.ipynb`
	- Hackathon notebooks for **day‑by‑day experimentation, training and analysis**.
- `J2/`
	- Additional notebooks and scripts, e.g. `train_and_save_models_jour2.py`, `train_models_v2.py` for training and exporting models used by the app.
- `misc/`
	- Alternative / experimental utilities (e.g. `streamlit_app_bis.py`, `utils.py`).

---

Main Components
---------------

### 1. Baseline CNN

Defined in `streamlit_app.py` as `BaselineCNN`, a simple convolutional neural network trained to classify images into:

- **Class 0**: Non‑Defective
- **Class 1**: Defective

The corresponding weights can be stored in `models/baseline_cnn.pth` and are loaded by the app if present.

### 2. Feature Extractors + Shallow Models

The app also uses a `FeatureExtractor` wrapper around **pre‑trained CNN backbones** from `torchvision` (e.g. ResNet50, VGG16, DenseNet121). These are used to extract fixed feature vectors, which are then fed into **shallow classifiers** such as:

- SVM
- Random Forest
- XGBoost

These shallow models, together with their scalers, are stored as `*.pkl` and `*_scaler.pkl` files in `models/`.

### 3. Ensemble Prediction

The Streamlit app aggregates predictions from:

- The baseline CNN.
- All available shallow models.

It then applies **majority voting** to output a final decision (Defective / Non‑Defective) with an overall confidence estimate and per‑model details.

If no trained models are found, the app can fall back to a **demo mode** with dummy models so that the UI remains usable during development.

---

Running the Streamlit App
-------------------------

1. **Install dependencies**

	 ```bash
	 pip install -r requirements.txt
	 ```

2. **(Optional) Place trained models**

	 - Put `baseline_cnn.pth` and any `*.pkl` / `*_scaler.pkl` files into the `models/` directory.

3. **Launch the app**

	 From the project root:

	 ```bash
	 streamlit run streamlit_app.py
	 ```

4. **Use the web interface**

	 - Open the URL shown by Streamlit (usually `http://localhost:8501`).
	 - In the sidebar, you can enable **Demo Mode** to use simulated predictions when no real models are available.
	 - Upload an image of an industrial component.
	 - Click **“Analyze Image”** to see:
		 - Final decision (Defective / Non‑Defective) with global confidence.
		 - Per‑model votes and confidences.
		 - Visualizations such as confidence gauges and vote distributions.

---

Using the Notebooks
-------------------

The notebooks (`01.1hackathon_notebook_template_jour1.ipynb`, `01.1hackathon_notebook_template_jour2.ipynb` and those in `J2/`) are organized to guide you through:

- Data exploration and basic preprocessing.
- Training and evaluating baseline and advanced models.
- Saving trained weights and classifiers into the `models/` folder for later use by the Streamlit app.

You can open them in Jupyter, VS Code, or any compatible notebook environment and run the cells step‑by‑step.

---

Goal of the Project
-------------------

The primary goal is to provide a **clear, hackathon‑friendly template** for:

- Building and experimenting with **image‑based defect detection models**.
- Quickly wrapping those models in a **user‑friendly Streamlit interface**.
- Demonstrating how to combine **deep feature extractors** and **classical machine‑learning models** in an ensemble for robust predictions.

