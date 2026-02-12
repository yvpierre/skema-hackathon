
"""
🏭 Application Complète - Prédiction + CBIR
============================================
Application Streamlit pour:
1. Prédiction de défauts avec vote majoritaire (CNN + Shallow classifiers)
2. Recherche d'images similaires (CBIR)

Usage:
    streamlit run streamlit_app.py

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

# VLM Configuration
VLM_LANGUAGE = "fr"  # 'fr' ou 'en'


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           PAGE CONFIG                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="🏭 Détection de Défauts + CBIR + VLM",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé - Modern Glassmorphism UI (Dark Edition)
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════════
       GLOBAL STYLES & VARIABLES
    ═══════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary: #ffffff;
        --primary-light: #f1f5f9;
        --accent: #a1a1aa;
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #eab308;
        --glass-bg: rgba(255, 255, 255, 0.08);
        --glass-border: rgba(255, 255, 255, 0.12);
        --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        --shadow-lg: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    
    .stApp {
        background: linear-gradient(145deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #1a1a2e 75%, #0a0a0a 100%);
        background-attachment: fixed;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       GLASSMORPHISM CARDS
    ═══════════════════════════════════════════════════════════════════ */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .glass-card h3 {
        color: #ffffff !important;
    }
    
    .glass-card-dark {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 2rem;
        color: white;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       HEADER STYLES
    ═══════════════════════════════════════════════════════════════════ */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       SIDEBAR STYLES - Right side, fixed 120px
    ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.85) !important;
        backdrop-filter: blur(20px);
        border-left: 1px solid rgba(255, 255, 255, 0.08);
        border-right: none;
        right: 0;
        left: auto !important;
        width: 120px !important;
        min-width: 120px !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 120px !important;
        padding: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e4e4e7 !important;
        font-size: 0.65rem !important;
    }
    
    [data-testid="stSidebar"] .stCheckbox label span {
        color: #e4e4e7 !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Sidebar toggle button */
    [data-testid="stSidebarCollapseButton"] {
        right: 0 !important;
        left: auto !important;
    }
    
    [data-testid="stSidebar"] .stSlider {
        padding: 0 !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       BUTTON STYLES - Glassy with white font
    ═══════════════════════════════════════════════════════════════════ */
    .stButton > button {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(10px);
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button span {
        color: #ffffff !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       IMAGE UPLOAD & DISPLAY
    ═══════════════════════════════════════════════════════════════════ */
    .upload-zone {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        max-width: 450px;
        margin: 0 auto;
    }
    
    .upload-zone:hover {
        border-color: rgba(255, 255, 255, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    
    .image-frame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    .image-frame img {
        border-radius: 8px;
        width: 100%;
    }
    
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       FILE UPLOADER - Bigger with colorful button
    ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 2px dashed rgba(138, 43, 226, 0.4);
        max-width: 450px;
        margin: 0 auto;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(138, 43, 226, 0.6);
        background: rgba(138, 43, 226, 0.08);
    }
    
    [data-testid="stFileUploader"] * {
        color: #e4e4e7 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 50%, #d946ef 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
        transform: translateY(-1px);
    }
    
    [data-testid="stFileUploader"] section {
        padding: 1rem !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       RESULT CARDS
    ═══════════════════════════════════════════════════════════════════ */
    .result-defective {
        background: rgba(239, 68, 68, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-left: 4px solid #ef4444;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.1);
    }
    
    .result-defective h2, .result-defective p {
        color: #fecaca !important;
    }
    
    .result-ok {
        background: rgba(34, 197, 94, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-left: 4px solid #22c55e;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(34, 197, 94, 0.1);
    }
    
    .result-ok h2, .result-ok p {
        color: #bbf7d0 !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       METRICS & STATUS CARDS
    ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 0.75rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    [data-testid="stMetric"] label {
        color: #71717a !important;
        font-weight: 500;
        font-size: 0.75rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 1rem !important;
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.75rem;
        backdrop-filter: blur(10px);
    }
    
    .status-active {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .status-inactive {
        background: rgba(161, 161, 170, 0.2);
        color: #a1a1aa;
        border: 1px solid rgba(161, 161, 170, 0.3);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       LEFT NAVIGATION PANEL (after image upload)
    ═══════════════════════════════════════════════════════════════════ */
    .nav-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 1rem;
        height: fit-content;
    }
    
    .nav-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.875rem 1rem;
        border-radius: 10px;
        color: #a1a1aa;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 0.5rem;
        border: 1px solid transparent;
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.05);
        color: #ffffff;
    }
    
    .nav-item.active {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    .nav-icon {
        font-size: 1.25rem;
        width: 24px;
        text-align: center;
    }
    
    .nav-label {
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       CENTERED HERO SECTION (smaller)
    ═══════════════════════════════════════════════════════════════════ */
    .hero-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 40vh;
        text-align: center;
        padding: 1rem;
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #71717a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        max-width: 400px;
    }
    
    .hero-upload {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        transition: all 0.3s ease;
        cursor: pointer;
        max-width: 300px;
    }
    
    .hero-upload:hover {
        border-color: rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       MAIN BODY GLASS CONTAINER
    ═══════════════════════════════════════════════════════════════════ */
    .main-glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       SECTION CONTENT BOXES (for readability)
    ═══════════════════════════════════════════════════════════════════ */
    .section-content {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }
    
    .section-content h3, .section-content h4 {
        color: #ffffff !important;
        margin-bottom: 1rem;
    }
    
    .section-content p {
        color: #e4e4e7 !important;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       HORIZONTAL NAV BUTTONS
    ═══════════════════════════════════════════════════════════════════ */
    .nav-horizontal {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        padding: 0.75rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-bottom: 1rem;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       TABS STYLING
    ═══════════════════════════════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 0.5rem;
        gap: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        color: #a1a1aa;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       VLM & URGENCY BADGES
    ═══════════════════════════════════════════════════════════════════ */
    .vlm-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-left: 4px solid #a1a1aa;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
    }
    
    .urgency-urgent {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .urgency-high {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .urgency-moderate {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1e293b;
        padding: 0.5rem 1.25rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 4px 15px rgba(251, 191, 36, 0.4);
    }
    
    .urgency-none {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       CBIR IMAGE CARDS
    ═══════════════════════════════════════════════════════════════════ */
    .cbir-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .cbir-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .cbir-card-defective {
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    .cbir-card-ok {
        border-color: rgba(34, 197, 94, 0.4);
    }
    
    .cbir-card-content {
        padding: 1rem;
        text-align: center;
        color: #e4e4e7;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       MODEL CARDS
    ═══════════════════════════════════════════════════════════════════ */
    .model-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    }
    
    .model-card-defect {
        border-color: rgba(239, 68, 68, 0.3);
        background: rgba(239, 68, 68, 0.1);
    }
    
    .model-card-ok {
        border-color: rgba(34, 197, 94, 0.3);
        background: rgba(34, 197, 94, 0.1);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       PROGRESS BAR
    ═══════════════════════════════════════════════════════════════════ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ffffff 0%, #a1a1aa 100%);
        border-radius: 100px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       DATAFRAME STYLING
    ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ANIMATIONS
    ═══════════════════════════════════════════════════════════════════ */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       HIDE STREAMLIT BRANDING
    ═══════════════════════════════════════════════════════════════════ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ═══════════════════════════════════════════════════════════════════
       CUSTOM SCROLLBAR
    ═══════════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 100px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 100px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ADDITIONAL DARK MODE FIXES
    ═══════════════════════════════════════════════════════════════════ */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e4e4e7 !important;
    }
    
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #a1a1aa !important;
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
    
    .stAlert p {
        color: #e4e4e7 !important;
    }
    
    /* Plotly charts dark mode */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border-color: rgba(34, 197, 94, 0.3) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-color: rgba(239, 68, 68, 0.3) !important;
    }
    
    .stWarning {
        background: rgba(234, 179, 8, 0.1) !important;
        border-color: rgba(234, 179, 8, 0.3) !important;
    }
    
    .stInfo {
        background: rgba(255, 255, 255, 0.05) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
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
# ║                           VLM - VISION LANGUAGE MODEL                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TemplateVLM:
    """Générateur de descriptions basé sur templates (pas de GPU requis)."""
    
    def __init__(self, language='fr'):
        self.language = language
        
        self.templates = {
            'fr': {
                'defective': {
                    'high': {
                        'title': "⚠️ DÉFAUT CRITIQUE DÉTECTÉ",
                        'description': "L'analyse révèle une anomalie majeure sur cette pièce industrielle.",
                        'confidence_text': "La confiance du système est très élevée ({confidence:.1%}).",
                        'details': "**Type probable:** {defect_type}\n**Zone affectée:** {zone}",
                        'recommendation': "🚨 **ACTION IMMÉDIATE:** Retirer cette pièce de la production. Inspection manuelle obligatoire.",
                        'urgency': "URGENT",
                        'urgency_class': "urgency-urgent"
                    },
                    'medium': {
                        'title': "⚠️ DÉFAUT DÉTECTÉ",
                        'description': "L'analyse indique une anomalie sur cette pièce.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "**Type probable:** {defect_type}\n**Inspection recommandée**",
                        'recommendation': "📋 **ACTION:** Mettre en quarantaine pour inspection complémentaire.",
                        'urgency': "ÉLEVÉ",
                        'urgency_class': "urgency-high"
                    },
                    'low': {
                        'title': "⚠️ DÉFAUT POSSIBLE",
                        'description': "L'analyse suggère une anomalie potentielle.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Le système n'est pas certain. Vérification manuelle conseillée.",
                        'recommendation': "📝 **ACTION:** Vérification visuelle recommandée.",
                        'urgency': "MODÉRÉ",
                        'urgency_class': "urgency-moderate"
                    }
                },
                'non_defective': {
                    'high': {
                        'title': "✅ PIÈCE CONFORME",
                        'description': "L'analyse confirme que cette pièce ne présente aucun défaut visible.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Tous les critères de qualité sont satisfaits.",
                        'recommendation': "👍 Cette pièce peut continuer dans le processus de production.",
                        'urgency': "AUCUN",
                        'urgency_class': "urgency-none"
                    },
                    'medium': {
                        'title': "✅ PIÈCE PROBABLEMENT CONFORME",
                        'description': "L'analyse suggère que cette pièce est en bon état.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Les critères principaux sont satisfaits.",
                        'recommendation': "👍 Peut continuer, vérification optionnelle.",
                        'urgency': "AUCUN",
                        'urgency_class': "urgency-none"
                    },
                    'low': {
                        'title': "❓ STATUT INCERTAIN",
                        'description': "L'analyse n'est pas concluante.",
                        'confidence_text': "Confiance: {confidence:.1%}",
                        'details': "Le système hésite entre conforme et non-conforme.",
                        'recommendation': "🔍 Inspection manuelle recommandée.",
                        'urgency': "MODÉRÉ",
                        'urgency_class': "urgency-moderate"
                    }
                }
            },
            'en': {
                'defective': {
                    'high': {
                        'title': "⚠️ CRITICAL DEFECT DETECTED",
                        'description': "Analysis reveals a major anomaly on this industrial part.",
                        'confidence_text': "System confidence is very high ({confidence:.1%}).",
                        'details': "**Probable type:** {defect_type}\n**Affected zone:** {zone}",
                        'recommendation': "🚨 **IMMEDIATE ACTION:** Remove from production. Manual inspection required.",
                        'urgency': "URGENT",
                        'urgency_class': "urgency-urgent"
                    },
                    'medium': {
                        'title': "⚠️ DEFECT DETECTED",
                        'description': "Analysis indicates an anomaly on this part.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "**Probable type:** {defect_type}\n**Inspection recommended**",
                        'recommendation': "📋 **ACTION:** Quarantine for additional inspection.",
                        'urgency': "HIGH",
                        'urgency_class': "urgency-high"
                    },
                    'low': {
                        'title': "⚠️ POSSIBLE DEFECT",
                        'description': "Analysis suggests a potential anomaly.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "System is uncertain. Manual verification advised.",
                        'recommendation': "📝 **ACTION:** Visual verification recommended.",
                        'urgency': "MODERATE",
                        'urgency_class': "urgency-moderate"
                    }
                },
                'non_defective': {
                    'high': {
                        'title': "✅ PART CONFORMING",
                        'description': "Analysis confirms this part shows no visible defects.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "All quality criteria are satisfied.",
                        'recommendation': "👍 This part can continue in production.",
                        'urgency': "NONE",
                        'urgency_class': "urgency-none"
                    },
                    'medium': {
                        'title': "✅ PART LIKELY CONFORMING",
                        'description': "Analysis suggests this part is in good condition.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "Main criteria are satisfied.",
                        'recommendation': "👍 Can continue, optional verification.",
                        'urgency': "NONE",
                        'urgency_class': "urgency-none"
                    },
                    'low': {
                        'title': "❓ UNCERTAIN STATUS",
                        'description': "Analysis is inconclusive.",
                        'confidence_text': "Confidence: {confidence:.1%}",
                        'details': "System uncertain between conforming and non-conforming.",
                        'recommendation': "🔍 Manual inspection recommended.",
                        'urgency': "MODERATE",
                        'urgency_class': "urgency-moderate"
                    }
                }
            }
        }
        
        self.defect_types = {
            'fr': ["rayure", "fissure", "déformation", "corrosion", "inclusion", "porosité", "bavure"],
            'en': ["scratch", "crack", "deformation", "corrosion", "inclusion", "porosity", "burr"]
        }
        
        self.zones = {
            'fr': ["surface principale", "bord", "centre", "coin", "jonction"],
            'en': ["main surface", "edge", "center", "corner", "junction"]
        }
    
    def get_confidence_level(self, confidence):
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        return 'low'
    
    def generate(self, prediction, confidence, cbir_results=None):
        """Génère une description structurée."""
        lang = self.language
        class_name = 'defective' if prediction == 1 else 'non_defective'
        level = self.get_confidence_level(confidence)
        
        template = self.templates[lang][class_name][level]
        
        # Sélectionner défaut et zone basé sur confidence
        seed = int(confidence * 1000)
        defect_type = self.defect_types[lang][seed % len(self.defect_types[lang])]
        zone = self.zones[lang][seed % len(self.zones[lang])]
        
        # CBIR info
        cbir_text = ""
        if cbir_results and len(cbir_results) > 0:
            # Flatten CBIR results if nested by model
            all_results = []
            if isinstance(cbir_results, dict):
                for model_results in cbir_results.values():
                    all_results.extend(model_results)
            else:
                all_results = cbir_results
            
            if all_results:
                defect_count = sum(1 for r in all_results if r.get('label', 0) == 1)
                total = len(all_results)
                if lang == 'fr':
                    cbir_text = f"**Analyse CBIR:** {defect_count}/{total} images similaires sont défectueuses."
                else:
                    cbir_text = f"**CBIR Analysis:** {defect_count}/{total} similar images are defective."
        
        return {
            'title': template['title'],
            'description': template['description'],
            'confidence_text': template['confidence_text'].format(confidence=confidence),
            'details': template['details'].format(defect_type=defect_type, zone=zone),
            'recommendation': template['recommendation'],
            'urgency': template['urgency'],
            'urgency_class': template['urgency_class'],
            'cbir_text': cbir_text
        }


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
    cnn_path = MODELS_DIR / 'cnn_baseline.pth'
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
        if '_scaler' in pkl_file.stem or pkl_file.stem == 'scaler':
            # C'est un scaler - handle both 'scaler.pkl' and '*_scaler.pkl'
            if pkl_file.stem == 'scaler':
                # Generic scaler, associate with all extractors
                try:
                    scaler = joblib.load(pkl_file)
                    for ext_name in ['resnet50', 'vgg16', 'densenet121']:
                        models_dict['scalers'][ext_name] = scaler
                except:
                    pass
            else:
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


@st.cache_resource
def load_vlm():
    """Charge le générateur VLM."""
    return TemplateVLM(language=VLM_LANGUAGE)


def predict_ensemble(image_tensor, extractors, models_dict, use_cnn=True, use_resnet=True, use_vgg=True):
    """Prédiction avec vote majoritaire."""
    predictions = {}
    
    # CNN predictions
    if use_cnn:
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
        # Filter based on model selection
        if not use_resnet and 'resnet' in name.lower():
            continue
        if not use_vgg and 'vgg' in name.lower():
            continue
        
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
    
    # Weighted confidence: average confidence of models voting for final prediction
    confidences_for_pred = [
        p['confidence'] for p in predictions.values() 
        if p['prediction'] == final_pred
    ]
    
    if confidences_for_pred:
        # Use weighted average: sum of (confidence * confidence) / sum of confidence
        confidence_weights = sum(c * c for c in confidences_for_pred) / sum(confidences_for_pred)
    else:
        confidence_weights = 0.0
    
    return {
        'prediction': final_pred,
        'class_name': CLASSES[final_pred],
        'confidence': confidence_weights,
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
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'pred_result' not in st.session_state:
        st.session_state.pred_result = None
    if 'cbir_results' not in st.session_state:
        st.session_state.cbir_results = None
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "vlm"
    
    # Right Sidebar (Config) - Fixed 120px
    with st.sidebar:
        st.markdown('<p style="text-align:center; margin:0;">⚙️</p>', unsafe_allow_html=True)
        st.markdown("---")
        demo_mode = st.checkbox("Demo", value=False)
        st.markdown("---")
        st.markdown('<p style="opacity:0.6; margin:0 0 0.25rem 0;">MOD.</p>', unsafe_allow_html=True)
        use_cnn = st.checkbox("CNN", value=True)
        use_resnet = st.checkbox("ResNet", value=True)
        use_vgg = st.checkbox("VGG", value=True)
        st.markdown("---")
        st.markdown('<p style="opacity:0.6; margin:0 0 0.25rem 0;">CBIR</p>', unsafe_allow_html=True)
        k_results = st.slider("K", 1, 10, 5, label_visibility="collapsed")
        distance_metric = "Cosinus"  # Must match DISTANCE_FUNCTIONS key
        st.markdown("---")
        vlm_enabled = st.checkbox("VLM", value=True)
    
    # Load resources silently (no loading message)
    extractors = load_feature_extractors()
    models_dict = load_classification_models()
    signatures = load_cbir_signatures()
    vlm = load_vlm()
    
    # ════════════════════════════════════════════════════════════════════
    # PERMANENT HEADER - Always visible
    # ════════════════════════════════════════════════════════════════════
    st.markdown('''
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <h1 class="main-header">🏭 Détection de Défauts</h1>
        <p class="sub-header">Classification IA • Recherche d'images similaires • Description VLM</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # File uploader (below title)
    uploaded_file = st.file_uploader("upload", type=['jpg', 'jpeg', 'png', 'bmp'], label_visibility="collapsed", key="main_uploader")
    
    # ════════════════════════════════════════════════════════════════════
    # INITIAL STATE: Instruction text
    # ════════════════════════════════════════════════════════════════════
    if not uploaded_file:
        st.markdown('''
        <div style="text-align: center; color: #71717a; font-size: 0.9rem; padding: 2rem 0;">
            <p>👆 Glissez une image ci-dessus ou cliquez pour sélectionner</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # ════════════════════════════════════════════════════════════════════
    # AFTER UPLOAD: Image on left, content on right
    # ════════════════════════════════════════════════════════════════════
    else:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Layout: Image (1) | Content (2)
        img_col, content_col = st.columns([1, 2])
        
        with img_col:
            # Image preview with glass frame
            st.markdown('<div class="main-glass-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
            # Compact metrics under image
            n_classifiers = len(models_dict.get('cnn_models', {})) + len(models_dict.get('shallow_models', {}))
            st.markdown(f'''
            <div style="display: flex; justify-content: space-around; margin-top: 1rem; font-size: 0.75rem; color: #a1a1aa;">
                <span>🎯 {n_classifiers if n_classifiers > 0 else "Demo"}</span>
                <span>🧠 {len(extractors)}</span>
                <span>📚 {len(signatures) if signatures else "—"}</span>
                <span>🤖 {"✓" if vlm_enabled else "✗"}</span>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with content_col:
            # Analysis button or results
            if not st.session_state.analysis_done:
                st.markdown('<div class="main-glass-container">', unsafe_allow_html=True)
                st.markdown("### 🔬 Prêt pour l'analyse")
                st.markdown("Cliquez pour lancer la classification et la recherche d'images similaires.")
                if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
                    transform = get_transform()
                    image_tensor = transform(image).unsqueeze(0)
                    
                    progress = st.progress(0)
                    status = st.empty()
                    
                    status.text("🎯 Classification...")
                    progress.progress(30)
                    
                    if demo_mode:
                        time.sleep(0.5)
                        pred_result = {
                            'prediction': np.random.choice([0, 1]),
                            'class_name': np.random.choice(['non_defective', 'defective']),
                            'confidence': np.random.uniform(0.6, 0.95),
                            'votes': {'defective': np.random.randint(2, 5), 'non_defective': np.random.randint(1, 3)},
                            'model_results': {
                                'CNN': {'prediction': 1, 'confidence': 0.85},
                                'ResNet': {'prediction': 1, 'confidence': 0.90},
                                'VGG': {'prediction': 0, 'confidence': 0.75},
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
                        pred_result = predict_ensemble(image_tensor, extractors, models_dict, 
                                                      use_cnn=use_cnn, use_resnet=use_resnet, use_vgg=use_vgg)
                    
                    progress.progress(60)
                    
                    status.text("🔍 Recherche CBIR...")
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
                                except:
                                    pass
                    elif demo_mode:
                        for model_name in ['resnet50', 'vgg16']:
                            cbir_results[model_name] = [
                                {
                                    'rank': i+1,
                                    'path': f'./data/train/defective/img_{np.random.randint(1, 100):03d}.jpg',
                                    'distance': np.random.uniform(0.1, 0.5),
                                    'label': np.random.choice([0, 1]),
                                }
                                for i in range(k_results)
                            ]
                    
                    progress.progress(100)
                    status.empty()
                    progress.empty()
                    
                    st.session_state.pred_result = pred_result
                    st.session_state.cbir_results = cbir_results
                    st.session_state.analysis_done = True
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:
                pred_result = st.session_state.pred_result
                cbir_results = st.session_state.cbir_results
                
                # Result header (prediction result)
                if pred_result:
                    if pred_result['prediction'] == 1:
                        st.markdown(f'''
                        <div class="result-defective">
                            <h2 style="margin: 0;">⚠️ DÉFAUT DÉTECTÉ</h2>
                            <p style="margin: 0.5rem 0 0 0;">Confiance: <strong>{pred_result['confidence']:.1%}</strong>
                            ({pred_result['votes']['defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="result-ok">
                            <h2 style="margin: 0;">✅ PIÈCE CONFORME</h2>
                            <p style="margin: 0.5rem 0 0 0;">Confiance: <strong>{pred_result['confidence']:.1%}</strong>
                            ({pred_result['votes']['non_defective']}/{sum(pred_result['votes'].values())} modèles)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Horizontal navigation buttons
                nav_items = [
                    ("🤖", "VLM", "vlm"),
                    ("📊", "Vote", "vote"),
                    ("🎯", "Modèles", "models"),
                    ("🔍", "CBIR", "cbir"),
                    ("📈", "Détails", "details"),
                ]
                
                nav_cols = st.columns(len(nav_items))
                for i, (icon, label, key) in enumerate(nav_items):
                    with nav_cols[i]:
                        if st.button(f"{icon} {label}", key=f"nav_{key}", use_container_width=True):
                            st.session_state.current_tab = key
                            st.rerun()
                
                # Section content with background
                current_tab = st.session_state.current_tab
                
                st.markdown('<div class="section-content">', unsafe_allow_html=True)
                
                if current_tab == "vlm":
                    st.markdown("### 🤖 Description VLM")
                    if vlm_enabled and pred_result:
                        vlm_result = vlm.generate(
                            prediction=pred_result['prediction'],
                            confidence=pred_result['confidence'],
                            cbir_results=cbir_results
                        )
                        st.markdown(f"#### {vlm_result['title']}")
                        st.markdown(f'<span class="{vlm_result["urgency_class"]}">Urgence: {vlm_result["urgency"]}</span>', unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown(f"**{vlm_result['description']}**")
                        st.markdown(vlm_result['confidence_text'])
                        st.markdown(vlm_result['details'])
                        if vlm_result['cbir_text']:
                            st.markdown(vlm_result['cbir_text'])
                        st.info(vlm_result['recommendation'])
                    else:
                        st.info("VLM désactivé")
                
                elif current_tab == "vote":
                    st.markdown("### 📊 Vote Majoritaire")
                    if pred_result and pred_result.get('model_results'):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            fig = px.pie(
                                values=[pred_result['votes']['non_defective'], pred_result['votes']['defective']],
                                names=['OK', 'Défaut'],
                                color_discrete_sequence=['#22c55e', '#ef4444'],
                                hole=0.4
                            )
                            fig.update_layout(
                                height=250,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color='#e4e4e7'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with col_b:
                            model_data = []
                            for name, res in pred_result['model_results'].items():
                                model_data.append({
                                    'Modèle': name,
                                    'Vote': '🔴 Défaut' if res['prediction'] == 1 else '🟢 OK',
                                    'Conf.': f"{res['confidence']:.0%}"
                                })
                            st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
                
                elif current_tab == "models":
                    st.markdown("### 🎯 Analyse Modèles")
                    if pred_result and pred_result.get('model_results'):
                        cols = st.columns(min(len(pred_result['model_results']), 4))
                        for idx, (name, res) in enumerate(pred_result['model_results'].items()):
                            with cols[idx % len(cols)]:
                                color = 'rgba(239,68,68,0.15)' if res['prediction'] == 1 else 'rgba(34,197,94,0.15)'
                                border = 'rgba(239,68,68,0.4)' if res['prediction'] == 1 else 'rgba(34,197,94,0.4)'
                                icon = '🔴' if res['prediction'] == 1 else '🟢'
                                st.markdown(f'''
                                <div style="background: {color}; border: 1px solid {border}; border-radius: 10px; padding: 0.75rem; text-align: center;">
                                    <p style="margin: 0; font-size: 0.75rem; opacity: 0.7;">{name}</p>
                                    <p style="margin: 0.25rem 0; font-size: 1.25rem;">{icon}</p>
                                    <p style="margin: 0; font-weight: 600; font-size: 0.9rem;">{res['confidence']:.1%}</p>
                                </div>
                                ''', unsafe_allow_html=True)
                
                elif current_tab == "cbir":
                    st.markdown("### 🔍 Images Similaires (CBIR)")
                    # Debug info
                    st.caption(f"Signatures chargées: {list(cbir_results.keys()) if cbir_results else 'Aucune'}")
                    if cbir_results and len(cbir_results) > 0:
                        for model_name, results in cbir_results.items():
                            st.markdown(f"**Extracteur: {model_name.upper()}**")
                            if results and len(results) > 0:
                                num_cols = min(len(results), 5)
                                cols = st.columns(num_cols)
                                for i, res in enumerate(results[:num_cols]):
                                    with cols[i]:
                                        img_path = Path(res['path'])
                                        if img_path.exists():
                                            try:
                                                st.image(Image.open(img_path), use_container_width=True, caption=f"#{res['rank']}")
                                            except:
                                                st.markdown(f"📷 Image #{res['rank']}")
                                        else:
                                            st.markdown(f"📷 Image #{res['rank']}")
                                        label_icon = "🔴 Défaut" if res['label'] == 1 else "🟢 OK"
                                        st.markdown(f"<p style='text-align:center; font-size:0.7rem;'>{label_icon}<br>dist: {res['distance']:.3f}</p>", unsafe_allow_html=True)
                            else:
                                st.info(f"Aucun résultat pour {model_name}")
                            st.markdown("---")
                    else:
                        st.warning("⚠️ Aucun résultat CBIR. Vérifiez que l'analyse a été effectuée.")
                        st.caption(f"Signatures disponibles: {len(signatures)} | Extracteurs: {list(extractors.keys())}")
                
                elif current_tab == "details":
                    st.markdown("### 📈 Détails")
                    if pred_result:
                        fig = create_gauge_chart(pred_result['confidence'], "Confiance")
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e4e4e7')
                        st.plotly_chart(fig, use_container_width=True)
                        if pred_result['confidence'] >= 0.8:
                            st.success("✅ Haute confiance - Modèles en accord")
                        elif pred_result['confidence'] >= 0.6:
                            st.warning("⚠️ Confiance moyenne - Désaccord partiel")
                        else:
                            st.error("❌ Faible confiance - Vérification manuelle")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Reset button
                if st.button("🔄 Nouvelle analyse", use_container_width=True):
                    st.session_state.analysis_done = False
                    st.session_state.pred_result = None
                    st.session_state.cbir_results = None
                    st.rerun()
    
    # Footer
    st.markdown('''
    <div style="text-align: center; padding: 0.5rem;">
        <p style="color: rgba(255,255,255,0.3); font-size: 0.7rem; margin: 0;">
            🏭 Hackathon IA — Classification • CBIR • VLM
        </p>
    </div>
    ''', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
