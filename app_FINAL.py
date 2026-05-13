"""
Brain Tumor Classification App - PRODUCTION READY WITH LOGIN
Using GLCM Features + Random Forest Classifier

Author: JIHAN SUCI ANANDA
Date: 2025
Version: 2.0 FINAL - WITH AUTHENTICATION
"""

import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import hashlib

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE INITIALIZATION ====================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'show_developer_page' not in st.session_state:
    st.session_state.show_developer_page = False

# ==================== USER CREDENTIALS ====================
# Format: username: password_hash
USERS = {
    "jihan@gmail.com": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # password: admin123
    "glcm@gmail.com": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  # password: admin
}

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username, password):
    """Verify username and password"""
    if username in USERS:
        return USERS[username] == hash_password(password)
    return False

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Login Container */
    .login-container {
        background: linear-gradient(135deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(0, 255, 255, 0.3);
        border: 2px solid rgba(0, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 500px;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 20px 60px rgba(0, 255, 255, 0.3);
        }
        to {
            box-shadow: 0 20px 60px rgba(138, 43, 226, 0.5);
        }
    }
    
    .login-title {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8),
                     0 0 40px rgba(0, 255, 255, 0.6),
                     0 0 60px rgba(0, 255, 255, 0.4);
        letter-spacing: 3px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.8),
                         0 0 40px rgba(0, 255, 255, 0.6),
                         0 0 60px rgba(0, 255, 255, 0.4);
        }
        50% {
            text-shadow: 0 0 30px rgba(138, 43, 226, 0.8),
                         0 0 50px rgba(138, 43, 226, 0.6),
                         0 0 70px rgba(138, 43, 226, 0.4);
        }
    }
    
    .login-subtitle {
        font-family: 'Rajdhani', sans-serif;
        color: #a0a0ff;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 2px;
    }
    
    .brain-scan-container {
        text-align: center;
        margin: 2rem 0;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(138, 43, 226, 0.2) 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 255, 255, 0.3);
        border: 2px solid rgba(0, 255, 255, 0.4);
        backdrop-filter: blur(10px);
    }
    
    .header-title {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8),
                     0 0 40px rgba(0, 255, 255, 0.6);
        letter-spacing: 2px;
    }
    
    .header-subtitle {
        font-family: 'Rajdhani', sans-serif;
        color: #a0a0ff;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    .welcome-badge {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.3) 0%, rgba(138, 43, 226, 0.3) 100%);
        color: #00ffff;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        display: inline-block;
        border: 1px solid rgba(0, 255, 255, 0.5);
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, rgba(15,12,41,0.9) 0%, rgba(48,43,99,0.9) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 255, 255, 0.2);
        margin: 1rem 0;
        border-left: 5px solid #00ffff;
        border: 1px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%);
        color: white;
        font-weight: bold;
        font-family: 'Rajdhani', sans-serif;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 15px;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.4);
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.6);
        background: linear-gradient(135deg, #8a2be2 0%, #00ffff 100%);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        background: rgba(15,12,41,0.8);
        color: #00ffff;
        border: 2px solid rgba(0, 255, 255, 0.3);
        border-radius: 10px;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #00ffff;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(138, 43, 226, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 255, 255, 0.3);
        border: 2px solid rgba(0, 255, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 255, 255, 0.5);
    }
    
    /* Result badge */
    .result-badge {
        display: inline-block;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-family: 'Rajdhani', sans-serif;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .badge-glioma {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%);
        color: white;
        border: 2px solid #ff006e;
    }
    
    .badge-meningioma {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%);
        color: white;
        border: 2px solid #00b4d8;
    }
    
    .badge-no-tumor {
        background: linear-gradient(135deg, #06ffa5 0%, #00d9ff 100%);
        color: white;
        border: 2px solid #06ffa5;
    }
    
    .badge-pituitary {
        background: linear-gradient(135deg, #ffd60a 0%, #ff9500 100%);
        color: white;
        border: 2px solid #ffd60a;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
        border-right: 2px solid rgba(0, 255, 255, 0.3);
    }
    
    /* Info box styling */
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        border-left: 5px solid #00ffff;
        color: #a0a0ff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ffff 0%, #8a2be2 100%);
    }
    
    /* Tech lines decoration */
    .tech-line {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #00ffff 50%, transparent 100%);
        margin: 1rem 0;
        animation: scan 2s linear infinite;
    }
    
    @keyframes scan {
        0% {
            opacity: 0.3;
        }
        50% {
            opacity: 1;
        }
        100% {
            opacity: 0.3;
        }
    }
    
    /* Logout button */
    .logout-btn {
        background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .logout-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 0, 110, 0.5);
    }
    
    /* Developer Profile Page */
    .dev-profile-container {
        background: linear-gradient(135deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 20px 60px rgba(0, 255, 255, 0.3);
        border: 2px solid rgba(0, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        margin: 2rem auto;
        max-width: 800px;
    }
    
    .dev-profile-title {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
        letter-spacing: 2px;
    }
    
    .dev-photo-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    .dev-photo {
        border-radius: 50%;
        border: 5px solid #00ffff;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6),
                    0 0 60px rgba(138, 43, 226, 0.4);
        width: 200px;
        height: 200px;
        object-fit: cover;
        animation: photoGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes photoGlow {
        from {
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6),
                        0 0 60px rgba(138, 43, 226, 0.4);
        }
        to {
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.8),
                        0 0 80px rgba(138, 43, 226, 0.6);
        }
    }
    
    .dev-info-section {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 255, 255, 0.3);
        margin: 1rem 0;
    }
    
    .dev-info-label {
        font-family: 'Rajdhani', sans-serif;
        color: #00ffff;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .dev-info-value {
        font-family: 'Rajdhani', sans-serif;
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 400;
    }
    
    .dev-bio {
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0ff;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: justify;
        margin: 1.5rem 0;
    }
    
    .clickable-link {
        color: #00ffff;
        cursor: pointer;
        text-decoration: underline;
        transition: all 0.3s ease;
    }
    
    .clickable-link:hover {
        color: #8a2be2;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

def resolve_model_path(model_path):
    """
    Resolve model path — coba beberapa lokasi umum:
    1. Path persis yang diberikan
    2. Relatif terhadap direktori script ini (root repo)
    3. Flat: nama file saja di root repo (semua file di folder yg sama)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        model_path,
        os.path.join(base_dir, model_path),
        os.path.join(base_dir, os.path.basename(model_path)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@st.cache_resource
def load_model(model_path):
    """Load model — auto-detect path, support pickle"""
    resolved = resolve_model_path(model_path)
    if resolved is None:
        return None
    try:
        with open(resolved, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Gagal load model: {str(e)}")
        return None


def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extract GLCM features from an image"""
    try:
        img = cv2.resize(image, (128, 128))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        
        glcm = graycomatrix(img, distances=distances, angles=angles, 
                            levels=256, symmetric=True, normed=True)
        
        features = {}
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        
        for prop in properties:
            values = graycoprops(glcm, prop)
            features[f'{prop}_mean'] = np.mean(values)
            features[f'{prop}_std'] = np.std(values)
            features[f'{prop}_range'] = np.ptp(values)
        
        return features
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None


def predict_image(image, model):
    """Predict the class of a brain tumor image"""
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = extract_glcm_features(image)
        
        if features is None:
            return None, None, None
        
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction]  
        
        prob_dict = {class_names[i]: probabilities[i] for i in range(len(class_names))}
        
        return predicted_class, confidence, prob_dict
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None


def get_badge_class(class_name):
    """Get CSS class for result badge based on tumor type"""
    if 'glioma' in class_name.lower():
        return 'badge-glioma'
    elif 'meningioma' in class_name.lower():
        return 'badge-meningioma'
    elif 'no tumor' in class_name.lower():
        return 'badge-no-tumor'
    elif 'pituitary' in class_name.lower():
        return 'badge-pituitary'
    return 'badge-glioma'


def create_probability_chart(probabilities, title="Class Probabilities"):
    """Create an interactive bar chart for class probabilities"""
    df = pd.DataFrame({
        'Class': list(probabilities.keys()),
        'Probability': [p * 100 for p in probabilities.values()]
    })
    
    df = df.sort_values('Probability', ascending=True)
    
    colors = []
    for class_name in df['Class']:
        if 'glioma' in class_name.lower():
            colors.append('#ff006e')
        elif 'meningioma' in class_name.lower():
            colors.append('#00b4d8')
        elif 'no tumor' in class_name.lower():
            colors.append('#06ffa5')
        else:
            colors.append('#ffd60a')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Probability'],
            y=df['Class'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,255,255,0.5)', width=2)
            ),
            text=[f'{p:.1f}%' for p in df['Probability']],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00ffff', size=16)),
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,12,41,0.5)',
        font=dict(size=12, color='#a0a0ff'),
        showlegend=False
    )
    
    fig.update_xaxes(range=[0, 100], gridcolor='rgba(0,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(0,255,255,0.2)')
    
    return fig


# ==================== DEVELOPER PROFILE PAGE ====================
def developer_page():
    """Display developer profile page"""
    
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ BACK TO APP", use_container_width=True):
            st.session_state.show_developer_page = False
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Developer Profile Container
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        st.markdown("""
        <div class="dev-profile-container">
            <div class="dev-profile-title">👨‍💻 DEVELOPER PROFILE</div>
            <div class="tech-line"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Photo section
        st.markdown("<div class='dev-photo-container'>", unsafe_allow_html=True)
        
        # Try to load profile photo
        base_dir = os.path.dirname(os.path.abspath(__file__))
        profile_photo_path = None
        for ext in ['profil.jpeg', 'profil.jpg', 'profil.png']:
            candidate = os.path.join(base_dir, ext)
            if os.path.exists(candidate):
                profile_photo_path = candidate
                break

        if profile_photo_path:
            try:
                profile_img = Image.open(profile_photo_path)
                # Create circular image effect
                col_a, col_b, col_c = st.columns([1, 1, 1])
                with col_b:
                    st.image(profile_img, use_container_width=True)
            except Exception as e:
                st.warning("⚠️ Could not load profile photo")
                # Placeholder icon
                st.markdown("""
                <div style="text-align: center;">
                    <div style="width: 200px; height: 200px; margin: 0 auto; 
                                background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%);
                                border-radius: 50%; display: flex; align-items: center; 
                                justify-content: center; font-size: 5rem;">
                        👨‍💻
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Placeholder icon
            st.markdown("""
            <div style="text-align: center;">
                <div style="width: 200px; height: 200px; margin: 0 auto; 
                            background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%);
                            border-radius: 50%; display: flex; align-items: center; 
                            justify-content: center; font-size: 5rem; border: 5px solid #00ffff;
                            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);">
                    👩‍💻
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Information sections
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">👤 FULL NAME</div>
            <div class="dev-info-value">JIHAN SUCI ANANDA</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">📍 PLACE & DATE OF BIRTH</div>
            <div class="dev-info-value">Sungai Penuh, 25 Januari 2002</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">💼 POSITION</div>
            <div class="dev-info-value">AI/ML Engineer & Data Scientist</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Bio section
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">📝 ABOUT ME</div>
            <div class="dev-bio">
                Saya adalah seorang AI/ML Engineer yang passionate dalam mengembangkan solusi berbasis 
                kecerdasan buatan untuk menyelesaikan masalah kompleks di bidang kesehatan dan teknologi. 
                Dengan pengalaman dalam pengembangan sistem deteksi tumor otak menggunakan machine learning, 
                saya berkomitmen untuk menciptakan aplikasi yang dapat membantu tenaga medis dalam proses 
                diagnosis yang lebih akurat dan efisien. Keahlian saya mencakup Computer Vision, Deep Learning, 
                dan pengembangan aplikasi berbasis web yang user-friendly. Saya percaya bahwa teknologi AI 
                dapat memberikan dampak positif yang signifikan dalam meningkatkan kualitas layanan kesehatan 
                di Indonesia dan dunia.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Skills & Technologies
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">🛠️ TECHNICAL SKILLS</div>
            <div style="margin-top: 1rem;">
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Python
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Machine Learning
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Computer Vision
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    TensorFlow
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Scikit-learn
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Streamlit
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    OpenCV
                </span>
                <span style="display: inline-block; background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%); 
                             padding: 0.5rem 1rem; border-radius: 15px; margin: 0.3rem; color: white; font-family: Rajdhani;">
                    Data Analysis
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Contact info
        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">📧 CONTACT</div>
            <div style="margin-top: 1rem; font-family: Rajdhani; color: #a0a0ff; font-size: 1.1rem;">
                <p>📧 Email: jihan.suci@example.com</p>
                <p>🔗 LinkedIn: linkedin.com/in/jihansuci</p>
                <p>💻 GitHub: github.com/jihansuci</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Version info
        st.markdown(f"""
        <div style="text-align: center; color: #a0a0ff; font-family: Rajdhani; margin-top: 2rem;">
            <p>⚡ NeuroScan AI Version 2.0 - {datetime.now().year}</p>
            <p style="font-size: 0.9rem; opacity: 0.7;">Powered by Machine Learning & Computer Vision</p>
        </div>
        """, unsafe_allow_html=True)


# ==================== LOGIN PAGE ====================
def login_page():
    """Display login page with futuristic design"""
    
    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-title">NEUROSCAN AI</div>
            <div class="login-subtitle">Neural Network Brain Analysis System</div>
            <div class="tech-line"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Brain scan image/icon (using emoji as placeholder, you can replace with actual image)
        st.markdown("""
        <div class="brain-scan-container">
            <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="brainGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style="stop-color:#00ffff;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#8a2be2;stop-opacity:1" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
                        <feMerge>
                            <feMergeNode in="coloredBlur"/>
                            <feMergeNode in="SourceGraphic"/>
                        </feMerge>
                    </filter>
                </defs>
                <circle cx="100" cy="100" r="80" fill="none" stroke="url(#brainGradient)" stroke-width="3" filter="url(#glow)"/>
                <circle cx="100" cy="100" r="60" fill="none" stroke="url(#brainGradient)" stroke-width="2" opacity="0.6" filter="url(#glow)"/>
                <circle cx="100" cy="100" r="40" fill="none" stroke="url(#brainGradient)" stroke-width="2" opacity="0.4" filter="url(#glow)"/>
                <path d="M 60 100 Q 80 80, 100 100 T 140 100" fill="none" stroke="url(#brainGradient)" stroke-width="3" filter="url(#glow)"/>
                <path d="M 70 120 Q 85 110, 100 120 T 130 120" fill="none" stroke="url(#brainGradient)" stroke-width="2" filter="url(#glow)"/>
                <circle cx="100" cy="100" r="5" fill="#00ffff" filter="url(#glow)"/>
                <circle cx="70" cy="90" r="3" fill="#00ffff" opacity="0.8"/>
                <circle cx="130" cy="90" r="3" fill="#00ffff" opacity="0.8"/>
                <circle cx="85" cy="110" r="3" fill="#8a2be2" opacity="0.8"/>
                <circle cx="115" cy="110" r="3" fill="#8a2be2" opacity="0.8"/>
            </svg>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            st.markdown("<h3 style='color: #00ffff; text-align: center; font-family: Rajdhani;'>🔐 SECURE ACCESS</h3>", unsafe_allow_html=True)
            
            username = st.text_input("👤 USERNAME", placeholder="Enter your username", key="username_input")
            password = st.text_input("🔑 PASSWORD", type="password", placeholder="Enter your password", key="password_input")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button("LOGIN", use_container_width=True)
            
            if submit:
                if username and password:
                    if verify_credentials(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password!")
                else:
                    st.warning("⚠️ Please enter both username and password!")
        
        # Info box
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔒 **Secure Login Required** | This system uses advanced AI for medical diagnosis")
        
        # Default credentials hint (remove in production)
        with st.expander("🔑 Default Credentials (For Testing)"):
            st.markdown("""
           
            
            *⚠️ Change these credentials in production!*
            """)


# ==================== MAIN APP ====================
def main_app():
    """Main application after login"""
    
    # Header with welcome message
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"""
        <div class="header-container">
            <h1 class="header-title">🧠 NEUROSCAN AI</h1>
            <p class="header-subtitle">Advanced Brain Tumor Detection System | GLCM Features + Random Forest</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align: center; margin-top: 1rem;">
            <div class="welcome-badge">👤 {st.session_state.username.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()
    
    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ SYSTEM SETTINGS")

        # ── Auto-detect semua file .sav di repo ──────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sav_files = []
        for root, dirs, files in os.walk(base_dir):
            # skip folder python cache
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.venv']]
            for f in files:
                if f.endswith('.sav') or f.endswith('.pkl'):
                    rel = os.path.relpath(os.path.join(root, f), base_dir)
                    sav_files.append(rel)

        if sav_files:
            # Default ke RF model kalau ada
            default_idx = 0
            for i, p in enumerate(sav_files):
                if 'rf' in p.lower() and 'best' not in p.lower():
                    default_idx = i
                    break
            model_path = st.selectbox(
                "Pilih Model",
                options=sav_files,
                index=default_idx,
                help="File .sav yang terdeteksi di repository"
            )
        else:
            st.warning("⚠️ Tidak ada file .sav ditemukan di repository.")
            model_path = st.text_input(
                "Model Path (manual)",
                value="brain_tumor_rf_glcm_model.sav",
                help="Masukkan nama file model secara manual"
            )
        
        st.markdown("---")
        
        st.markdown("### 📊 MODEL INFORMATION")
        st.info("""
        **Model:** Random Forest Classifier
        
        **Features:** GLCM (Gray-Level Co-occurrence Matrix)
        - Contrast, Dissimilarity, Homogeneity
        - Energy, Correlation, ASM
        
        **Classes:**
        - 🔴 Glioma Tumor
        - 🔵 Meningioma Tumor
        - 🟢 No Tumor
        - 🟡 Pituitary Tumor
        """)
        
        st.markdown("---")
        
        st.markdown("### 📖 HOW TO USE")
        st.markdown("""
        1. Upload brain MRI images
        2. Click **PREDICT** button
        3. View AI analysis results
        4. Download detailed report
        """)
        
        st.markdown("---")
        
        st.markdown("### 👨‍💻 DEVELOPER")
        
        # Clickable developer link
        if st.button("👤 JIHAN SUCI ANANDA", use_container_width=True, help="Click to view developer profile"):
            st.session_state.show_developer_page = True
            st.rerun()
        
        st.markdown(f"*Version 2.0 - {datetime.now().year}*")
    
    # Load model
    resolved_path = resolve_model_path(model_path)
    if resolved_path is None:
        st.error(f"❌ Model tidak ditemukan: `{model_path}`")
        st.info("💡 Pastikan file .sav sudah di-upload ke repository GitHub.")
        return

    model = load_model(model_path)
    if model is None:
        st.error("❌ Gagal load model. Coba pilih model lain di sidebar.")
        return

    with st.sidebar:
        st.success(f"✅ Model aktif:\n`{os.path.basename(resolved_path)}`")
    
    # File uploader
    st.markdown("### 📤 UPLOAD BRAIN MRI SCANS")
    uploaded_files = st.file_uploader(
        "Choose brain MRI images (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple images for batch analysis"
    )
    
    if uploaded_files:
        st.success(f"✅ **{len(uploaded_files)} image(s) uploaded successfully**")
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        st.markdown("### 🖼️ SCAN PREVIEW")
        
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 START AI ANALYSIS", use_container_width=True)
        
        if predict_button:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"⚡ Analyzing {uploaded_file.name}...")
                
                image = Image.open(uploaded_file)
                predicted_class, confidence, probabilities = predict_image(image, model)
                
                if predicted_class is not None:
                    results.append({
                        'filename': uploaded_file.name,
                        'image': image,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Analysis completed!")
            progress_bar.empty()
            
            if len(results) == 0:
                st.error("❌ No images were successfully processed.")
                return
            
            st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
            st.markdown("## 🎯 ANALYSIS RESULTS")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            class_counts = {}
            for result in results:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            with col1:
                glioma_count = class_counts.get('Glioma Tumor', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="margin:0; font-size:2.5rem; color:#ff006e;">{glioma_count}</h3>
                    <p style="margin:0; font-family: Rajdhani; letter-spacing: 1px;">GLIOMA</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                meningioma_count = class_counts.get('Meningioma Tumor', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="margin:0; font-size:2.5rem; color:#00b4d8;">{meningioma_count}</h3>
                    <p style="margin:0; font-family: Rajdhani; letter-spacing: 1px;">MENINGIOMA</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                no_tumor_count = class_counts.get('No Tumor', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="margin:0; font-size:2.5rem; color:#06ffa5;">{no_tumor_count}</h3>
                    <p style="margin:0; font-family: Rajdhani; letter-spacing: 1px;">NO TUMOR</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pituitary_count = class_counts.get('Pituitary Tumor', 0)
                st.markdown(f"""
                <div class="metric-container">
                    <h3 style="margin:0; font-size:2.5rem; color:#ffd60a;">{pituitary_count}</h3>
                    <p style="margin:0; font-family: Rajdhani; letter-spacing: 1px;">PITUITARY</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
            
            # Detailed results
            for idx, result in enumerate(results):
                st.markdown(f"### 📋 SCAN RESULT #{idx + 1}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(result['image'], use_container_width=True)
                
                with col2:
                    st.markdown(f"**📁 FILENAME:** `{result['filename']}`")
                    
                    badge_class = get_badge_class(result['predicted_class'])
                    st.markdown(f"""
                    <div>
                        <strong style="color: #00ffff;">🎯 DIAGNOSIS:</strong>
                        <span class="result-badge {badge_class}">{result['predicted_class'].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    confidence_percent = result['confidence'] * 100
                    st.markdown(f"**📊 CONFIDENCE LEVEL:** `{confidence_percent:.2f}%`")
                    st.progress(result['confidence'])
                    
                    fig = create_probability_chart(
                        result['probabilities'],
                        title=f"Probability Distribution - {result['filename']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
            
            # Download results
            st.markdown("### 💾 DOWNLOAD REPORT")
            
            df_results = pd.DataFrame([
                {
                    'Filename': r['filename'],
                    'Predicted Class': r['predicted_class'],
                    'Confidence (%)': f"{r['confidence'] * 100:.2f}",
                    'Glioma Prob (%)': f"{r['probabilities']['Glioma Tumor'] * 100:.2f}",
                    'Meningioma Prob (%)': f"{r['probabilities']['Meningioma Tumor'] * 100:.2f}",
                    'No Tumor Prob (%)': f"{r['probabilities']['No Tumor'] * 100:.2f}",
                    'Pituitary Prob (%)': f"{r['probabilities']['Pituitary Tumor'] * 100:.2f}",
                }
                for r in results
            ])
            
            csv = df_results.to_csv(index=False)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📥 DOWNLOAD ANALYSIS REPORT (CSV)",
                    data=csv,
                    file_name=f"neuroscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        st.info("👆 **UPLOAD MRI SCANS TO BEGIN ANALYSIS**")
        
        st.markdown("### 📌 SYSTEM CAPABILITIES")
        st.markdown("""
        The NeuroScan AI system can detect and classify:
        
        - **🔴 Glioma Tumor** - Malignant brain tumor originating from glial cells
        - **🔵 Meningioma Tumor** - Tumor forming in the meninges (brain/spinal cord membranes)
        - **🟢 No Tumor** - Healthy brain tissue with no abnormalities detected
        - **🟡 Pituitary Tumor** - Tumor in the pituitary gland affecting hormone production
        
        **Supported Formats:** PNG, JPG, JPEG
        """)


# ==================== RUN APP ====================
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        # Check if developer page should be shown
        if st.session_state.show_developer_page:
            developer_page()
        else:
            main_app()


if __name__ == "__main__":
    main()
