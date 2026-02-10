import streamlit as st
from PIL import Image
import numpy as np
import utils
import time
import random

# -------------------
# Configuration page
# -------------------
st.set_page_config(
    page_title="Détection de pièce défectueuse",
    page_icon="🔍",
    layout="centered"
)

# -------------------
# CSS pour mode nuit & dégradé titres
# -------------------
st.markdown(
    """
    <style>
    /* Fond général */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Titres avec dégradé bleu/vert */
    h1, h2, h3, h4, h5, h6 {
        background: -webkit-linear-gradient(45deg, #00ffcc, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Sidebar foncé */
    [data-testid="stSidebar"] {
        background-color: #1c1f29;
        color: #00ffcc;
    }

    /* Widgets sidebar */
    .stSidebar .stButton>button {
        background-color: #00aaff;
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stSidebar .stSelectbox>div>div {
        background-color: #0e1117;
        color: white !important;   /* <-- texte en blanc */
    }
    .stSidebar .stCheckbox>label {
        color: white !important;   /* <-- texte en blanc */
    }
    /* File uploader texte */
    .stFileUploader>label, .stFileUploader>div>label {
        color: white !important;   /* <-- "Déposez une image ici" en blanc */
    }
    /* Header Streamlit */
    header {
        background-color: #0e1117 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------
# Titre & description
# -------------------
st.title("🔍 Détection de pièce défectueuse")
st.write(
    """
    Bienvenue dans l'application de détection de défauts de pièces de turbines.  
    Ici vous pouvez **télécharger une image** et **vérifier si la pièce est conforme**.
    """
)

# -------------------
# Upload image
# -------------------
uploaded_file = st.file_uploader(
    "Déposez une image ici",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Affichage de l'image uploadée
    st.image(image, caption="Image uploadée", use_column_width=True)

    # -------------------
    # Bouton "Analyser"
    # -------------------
    if st.button("Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            time.sleep(2)  # effet visuel
            
            # Résultat mock
            prediction = random.choice(["✅ Pièce OK", "❌ Pièce défectueuse"])
            confidence = random.uniform(70, 99)

        # ---- PAGE DE RESULTAT ----
        st.markdown("## Résultat de l'analyse")
    
        # Image principale
        st.image(image, caption="Image uploadée", use_column_width=True)
    
        # Résultat et confiance
        if "Défectueuse" in prediction:
            st.error(f"{prediction} (Confiance : {confidence:.2f}%)")
        else:
            st.success(f"{prediction} (Confiance : {confidence:.2f}%)")

        st.markdown("---")
        st.markdown("### Exemples de défauts similaires :")

        # --- Images similaires (mock) ---
        # tu peux remplacer les chemins par tes vraies images de défaut
        similar_images = [
            "examples/defect1.jpg",
            "examples/defect2.jpg",
            "examples/defect3.jpg"
        ]

        cols = st.columns(len(similar_images))
        for col, img_path in zip(cols, similar_images):
            try:
                sim_img = Image.open(img_path).convert("RGB")
                col.image(sim_img, width=150)
            except:
                col.write("Image manquante")
                
# -------------------
# Sidebar design
# -------------------
st.sidebar.title("⚙️ Options")
st.sidebar.write("Paramètres simulés pour l'interface :")
st.sidebar.checkbox("Afficher les détails techniques", value=True)
st.sidebar.selectbox("Mode de visualisation", ["Standard", "Avancé"])

st.markdown(
    """
    <style>
    /* Header Streamlit */
    header {
        background-color: #0e1117 !important;
    }

    /* Sidebar titre et texte */
    [data-testid="stSidebar"] div p, 
    [data-testid="stSidebar"] .stCheckbox label, 
    [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
    }
    
    /* File uploader texte */
    .stFileUploader label, .stFileUploader>div>label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
