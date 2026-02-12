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
    /* Dégradé bleu-vert pour le titre principal uniquement */
    .title-gradient {
        background: -webkit-linear-gradient(45deg, #00ffcc, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Fond général */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Titres avec dégradé bleu/vert */
    h2, h3, h4, h5, h6 {
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
st.markdown('<h1 class="title-gradient">🔍 Détection de pièce défectueuse</h1>', unsafe_allow_html=True)
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
st.markdown(
    """
    <style>
    /* Bouton analyser custom */
    div.stButton > button:first-child {
        background-color: #00aaff;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.5em 1.5em;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #0088cc;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if st.button("Analyser l'image"):
    with st.spinner("Analyse en cours..."):
        time.sleep(2)  # effet visuel
        
        # Résultat mock
        prediction = random.choice(["✅ Pièce OK", "❌ Pièce défectueuse"])
        confidence = random.uniform(70, 99)

    # Affichage résultat : Verdict très visible
    if "défectueuse" in prediction.lower():
        verdict_text = "❌ Pièce défectueuse"
        verdict_color = "#ff5555"  # rouge
    else:
        verdict_text = "✅ Pièce OK"
        verdict_color = "#55ff55"  # vert

    # ici on force la couleur avec style inline
    st.markdown(f"<h1 style='color:{verdict_color};'>{verdict_text}</h1>", unsafe_allow_html=True)

    # Slider confiance stylé en HTML/CSS
    confidence_value = int(confidence)  # valeur simulée
    st.markdown("<h3 style='color:white;'>Confiance du résultat</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background-color:#333333; width:100%; height:20px; border-radius:10px;'>
        <div style='background: linear-gradient(to right, white {confidence_value}%, #555555 {confidence_value}%); height:20px; border-radius:10px;'>
        </div>
    </div>
    <p style='color:white;'>{confidence_value}%</p>
    """, unsafe_allow_html=True)          

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
