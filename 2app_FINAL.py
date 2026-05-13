"""
NeuroScan AI — Brain Tumor Classification App
PRODUCTION READY WITH LOGIN + HOME PAGE (News)

Author  : JIHAN SUCI ANANDA
Date    : 2025
Version : 3.0 FINAL
"""

import streamlit as st
import numpy as np
import cv2
import pickle
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import hashlib
import requests
import feedparser

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SESSION STATE ====================
if 'logged_in'           not in st.session_state: st.session_state.logged_in           = False
if 'username'            not in st.session_state: st.session_state.username            = ""
if 'current_page'        not in st.session_state: st.session_state.current_page        = "home"
if 'show_developer_page' not in st.session_state: st.session_state.show_developer_page = False

# ==================== CREDENTIALS ====================
USERS = {
    "jihan@gmail.com": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",
    "glcm@gmail.com":  "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",
}

def hash_password(p):         return hashlib.sha256(p.encode()).hexdigest()
def verify_credentials(u, p): return USERS.get(u) == hash_password(p)

# ==================== NEWS CONFIG ====================
GNEWS_API_KEY = "7080dd29ec8c97ee8fcd55adf2aa3817"
GNEWS_BASE    = "https://gnews.io/api/v4/search"
NEWS_QUERIES  = ["brain tumor", "glioma treatment", "meningioma", "neuro-oncology", "brain cancer research"]
RSS_FEEDS     = {
    "ScienceDaily — Brain Tumor": "https://www.sciencedaily.com/rss/health_medicine/brain_tumor.xml",
    "ScienceDaily — Health":      "https://www.sciencedaily.com/rss/top/health.xml",
}

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

.main { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }

.login-container {
    background: linear-gradient(135deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
    padding: 3rem; border-radius: 25px;
    box-shadow: 0 20px 60px rgba(0,255,255,0.3);
    border: 2px solid rgba(0,255,255,0.3); backdrop-filter: blur(10px);
    margin: 2rem auto; max-width: 500px;
    animation: glow 2s ease-in-out infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 20px 60px rgba(0,255,255,0.3); }
    to   { box-shadow: 0 20px 60px rgba(138,43,226,0.5); }
}
.login-title {
    font-family: 'Orbitron', sans-serif; color: #00ffff;
    font-size: 3.5rem; font-weight: 900; text-align: center;
    margin-bottom: 0.5rem; letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,255,255,0.8), 0 0 40px rgba(0,255,255,0.6);
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { text-shadow: 0 0 20px rgba(0,255,255,0.8), 0 0 40px rgba(0,255,255,0.6); }
    50%     { text-shadow: 0 0 30px rgba(138,43,226,0.8), 0 0 50px rgba(138,43,226,0.6); }
}
.login-subtitle {
    font-family: 'Rajdhani', sans-serif; color: #a0a0ff;
    font-size: 1.3rem; text-align: center; margin-bottom: 2rem;
    font-weight: 300; letter-spacing: 2px;
}
.brain-scan-container { text-align: center; margin: 2rem 0; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }

.header-container {
    background: linear-gradient(135deg, rgba(0,255,255,0.2) 0%, rgba(138,43,226,0.2) 100%);
    padding: 2.5rem; border-radius: 20px; margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,255,255,0.3);
    border: 2px solid rgba(0,255,255,0.4); backdrop-filter: blur(10px);
}
.header-title {
    font-family: 'Orbitron', sans-serif; color: #00ffff;
    font-size: 3.5rem; font-weight: 900; text-align: center; margin: 0;
    text-shadow: 0 0 20px rgba(0,255,255,0.8), 0 0 40px rgba(0,255,255,0.6);
    letter-spacing: 2px;
}
.header-subtitle {
    font-family: 'Rajdhani', sans-serif; color: #a0a0ff;
    font-size: 1.3rem; text-align: center; margin-top: 0.5rem;
    font-weight: 400; letter-spacing: 1px;
}
.welcome-badge {
    background: linear-gradient(135deg, rgba(0,255,255,0.3) 0%, rgba(138,43,226,0.3) 100%);
    color: #00ffff; padding: 0.5rem 1.5rem; border-radius: 20px;
    font-family: 'Rajdhani', sans-serif; font-weight: 600;
    display: inline-block; border: 1px solid rgba(0,255,255,0.5);
}
.metric-container {
    background: linear-gradient(135deg, rgba(0,255,255,0.2) 0%, rgba(138,43,226,0.2) 100%);
    padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
    box-shadow: 0 5px 15px rgba(0,255,255,0.3);
    border: 2px solid rgba(0,255,255,0.3); transition: all 0.3s ease;
}
.metric-container:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,255,255,0.5); }
.result-badge {
    display: inline-block; padding: 0.7rem 1.5rem; border-radius: 25px;
    font-weight: bold; font-family: 'Rajdhani', sans-serif;
    margin: 0.5rem 0; font-size: 1.1rem; letter-spacing: 1px;
}
.badge-glioma     { background: linear-gradient(135deg, #ff006e 0%, #8338ec 100%); color: white; border: 2px solid #ff006e; }
.badge-meningioma { background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); color: white; border: 2px solid #00b4d8; }
.badge-no-tumor   { background: linear-gradient(135deg, #06ffa5 0%, #00d9ff 100%); color: white; border: 2px solid #06ffa5; }
.badge-pituitary  { background: linear-gradient(135deg, #ffd60a 0%, #ff9500 100%); color: white; border: 2px solid #ffd60a; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
    border-right: 2px solid rgba(0,255,255,0.3);
}
.stButton>button {
    background: linear-gradient(135deg, #00ffff 0%, #8a2be2 100%);
    color: white; font-weight: bold; font-family: 'Rajdhani', sans-serif;
    border: none; padding: 0.75rem 2rem; border-radius: 15px;
    font-size: 1.1rem; transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(0,255,255,0.4); letter-spacing: 1px;
}
.stButton>button:hover {
    transform: translateY(-2px) scale(1.03);
    box-shadow: 0 8px 25px rgba(0,255,255,0.6);
    background: linear-gradient(135deg, #8a2be2 0%, #00ffff 100%);
}
.stProgress > div > div > div { background: linear-gradient(90deg, #00ffff 0%, #8a2be2 100%); }
.tech-line {
    height: 2px;
    background: linear-gradient(90deg, transparent 0%, #00ffff 50%, transparent 100%);
    margin: 1rem 0; animation: scan 2s linear infinite;
}
@keyframes scan { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }
.dev-profile-container {
    background: linear-gradient(135deg, rgba(15,12,41,0.95) 0%, rgba(48,43,99,0.95) 100%);
    padding: 3rem; border-radius: 25px;
    box-shadow: 0 20px 60px rgba(0,255,255,0.3);
    border: 2px solid rgba(0,255,255,0.3); backdrop-filter: blur(10px);
    margin: 2rem auto; max-width: 800px;
}
.dev-profile-title {
    font-family: 'Orbitron', sans-serif; color: #00ffff;
    font-size: 2.5rem; font-weight: 900; text-align: center;
    margin-bottom: 2rem; text-shadow: 0 0 20px rgba(0,255,255,0.8); letter-spacing: 2px;
}
.dev-info-section {
    background: linear-gradient(135deg, rgba(0,255,255,0.1) 0%, rgba(138,43,226,0.1) 100%);
    padding: 1.5rem; border-radius: 15px;
    border: 1px solid rgba(0,255,255,0.3); margin: 1rem 0;
}
.dev-info-label { font-family: 'Rajdhani', sans-serif; color: #00ffff; font-size: 1rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 0.5rem; }
.dev-info-value { font-family: 'Rajdhani', sans-serif; color: #ffffff; font-size: 1.3rem; }
.dev-bio { font-family: 'Rajdhani', sans-serif; color: #e0e0ff; font-size: 1.1rem; line-height: 1.8; text-align: justify; margin: 1rem 0; }
.stTextInput>div>div>input {
    background: rgba(15,12,41,0.8); color: #00ffff;
    border: 2px solid rgba(0,255,255,0.3); border-radius: 10px;
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; padding: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                   MODEL FUNCTIONS                        ║
# ╚══════════════════════════════════════════════════════════╝

def resolve_model_path(model_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        model_path,
        os.path.join(base_dir, model_path),
        os.path.join(base_dir, os.path.basename(model_path)),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


@st.cache_resource
def load_model(model_path):
    resolved = resolve_model_path(model_path)
    if resolved is None:
        return None
    try:
        with open(resolved, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None


def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    try:
        img = cv2.resize(image, (128, 128))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        features = {}
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            v = graycoprops(glcm, prop)
            features[f'{prop}_mean']  = np.mean(v)
            features[f'{prop}_std']   = np.std(v)
            features[f'{prop}_range'] = np.ptp(v)
        return features
    except Exception as e:
        st.error(f"Error fitur GLCM: {e}")
        return None


def predict_image(image, model):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = extract_glcm_features(image)
        if features is None:
            return None, None, None
        fa   = np.array(list(features.values())).reshape(1, -1)
        pred = model.predict(fa)[0]
        prob = model.predict_proba(fa)[0]
        cls  = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        return cls[pred], prob[pred], {cls[i]: prob[i] for i in range(4)}
    except Exception as e:
        st.error(f"Error prediksi: {e}")
        return None, None, None


def get_badge_class(cls):
    c = cls.lower()
    if 'glioma'     in c: return 'badge-glioma'
    if 'meningioma' in c: return 'badge-meningioma'
    if 'no tumor'   in c: return 'badge-no-tumor'
    return 'badge-pituitary'


def create_probability_chart(probabilities, title="Class Probabilities"):
    df = pd.DataFrame({'Class': list(probabilities.keys()),
                       'Prob': [p * 100 for p in probabilities.values()]})
    df = df.sort_values('Prob', ascending=True)
    color_map = {'Glioma Tumor': '#ff006e', 'Meningioma Tumor': '#00b4d8',
                 'No Tumor': '#06ffa5', 'Pituitary Tumor': '#ffd60a'}
    fig = go.Figure(go.Bar(
        x=df['Prob'], y=df['Class'], orientation='h',
        marker=dict(color=[color_map.get(c, '#00ffff') for c in df['Class']],
                    line=dict(color='rgba(0,255,255,0.5)', width=2)),
        text=[f'{p:.1f}%' for p in df['Prob']], textposition='outside',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color='#00ffff', size=16)),
        xaxis_title="Probability (%)", height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,12,41,0.5)',
        font=dict(size=12, color='#a0a0ff'), showlegend=False,
    )
    fig.update_xaxes(range=[0, 110], gridcolor='rgba(0,255,255,0.2)')
    fig.update_yaxes(gridcolor='rgba(0,255,255,0.2)')
    return fig


# ╔══════════════════════════════════════════════════════════╗
# ║                   NEWS FUNCTIONS                         ║
# ╚══════════════════════════════════════════════════════════╝

@st.cache_data(ttl=3600)
def fetch_gnews(query, max_results=6):
    try:
        r = requests.get(GNEWS_BASE, params={
            "q": query, "lang": "en", "max": max_results,
            "sortby": "publishedAt", "apikey": GNEWS_API_KEY,
        }, timeout=10)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_rss(feed_name, url, max_results=5):
    try:
        feed = feedparser.parse(url)
        return [{
            "title":       e.get("title", "No title"),
            "description": e.get("summary", "")[:300],
            "url":         e.get("link", "#"),
            "source":      {"name": feed_name},
            "publishedAt": e.get("published", ""),
        } for e in feed.entries[:max_results]]
    except Exception:
        return []


def format_date(s):
    if not s: return ""
    try:    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").strftime("%d %b %Y, %H:%M")
    except: return s[:25]


def render_news_card(article):
    title  = article.get("title", "No Title")
    desc   = article.get("description", "")
    url    = article.get("url", "#")
    source = article.get("source", {}).get("name", "Unknown")
    date   = format_date(article.get("publishedAt", ""))
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(15,12,41,0.85) 0%,rgba(48,43,99,0.85) 100%);
                border:1px solid rgba(0,255,255,0.2); border-left:4px solid #00ffff;
                border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="background:rgba(0,255,255,0.12);color:#00ffff;font-family:Rajdhani,sans-serif;
                         font-size:0.75rem;font-weight:600;letter-spacing:1px;
                         padding:2px 10px;border-radius:20px;border:1px solid rgba(0,255,255,0.3);">{source}</span>
            <span style="color:#6060a0;font-size:0.75rem;">{date}</span>
        </div>
        <h4 style="color:#e0e0ff;font-family:Rajdhani,sans-serif;font-size:1.05rem;
                   font-weight:600;margin:8px 0 6px 0;line-height:1.4;">{title}</h4>
        <p style="color:#a0a0c0;font-size:0.85rem;line-height:1.6;margin-bottom:10px;">
            {desc[:220]}{'...' if len(desc) > 220 else ''}
        </p>
        <a href="{url}" target="_blank" style="color:#00ffff;font-size:0.82rem;
           font-family:Rajdhani,sans-serif;font-weight:600;text-decoration:none;letter-spacing:0.5px;">
            Baca selengkapnya →
        </a>
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                       PAGES                              ║
# ╚══════════════════════════════════════════════════════════╝

def login_page():
    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div class="login-container">
            <div class="login-title">NEUROSCAN AI</div>
            <div class="login-subtitle">Neural Network Brain Analysis System</div>
            <div class="tech-line"></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="brain-scan-container">
          <svg width="180" height="180" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%"   style="stop-color:#00ffff;stop-opacity:1"/>
                <stop offset="100%" style="stop-color:#8a2be2;stop-opacity:1"/>
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="4" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            <circle cx="100" cy="100" r="80" fill="none" stroke="url(#bg)" stroke-width="3" filter="url(#glow)"/>
            <circle cx="100" cy="100" r="60" fill="none" stroke="url(#bg)" stroke-width="2" opacity="0.6" filter="url(#glow)"/>
            <circle cx="100" cy="100" r="40" fill="none" stroke="url(#bg)" stroke-width="2" opacity="0.4" filter="url(#glow)"/>
            <path d="M60 100 Q80 80,100 100 T140 100" fill="none" stroke="url(#bg)" stroke-width="3" filter="url(#glow)"/>
            <path d="M70 120 Q85 110,100 120 T130 120" fill="none" stroke="url(#bg)" stroke-width="2" filter="url(#glow)"/>
            <circle cx="100" cy="100" r="5" fill="#00ffff" filter="url(#glow)"/>
            <circle cx="70"  cy="90"  r="3" fill="#00ffff" opacity="0.8"/>
            <circle cx="130" cy="90"  r="3" fill="#00ffff" opacity="0.8"/>
            <circle cx="85"  cy="110" r="3" fill="#8a2be2" opacity="0.8"/>
            <circle cx="115" cy="110" r="3" fill="#8a2be2" opacity="0.8"/>
          </svg>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("<h3 style='color:#00ffff;text-align:center;font-family:Rajdhani;'>🔐 SECURE ACCESS</h3>", unsafe_allow_html=True)
            username = st.text_input("👤 USERNAME", placeholder="Masukkan username")
            password = st.text_input("🔑 PASSWORD", type="password", placeholder="Masukkan password")
            st.markdown("<br>", unsafe_allow_html=True)
            _, c2, _ = st.columns([1, 2, 1])
            with c2:
                submit = st.form_submit_button("LOGIN", use_container_width=True)
            if submit:
                if username and password:
                    if verify_credentials(username, password):
                        st.session_state.logged_in    = True
                        st.session_state.username     = username
                        st.session_state.current_page = "home"
                        st.success("✅ Login berhasil!")
                        st.rerun()
                    else:
                        st.error("❌ Username atau password salah!")
                else:
                    st.warning("⚠️ Lengkapi username dan password!")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("🔒 **Secure Login Required** | Sistem AI untuk diagnosis medis")
        with st.expander("🔑 Kredensial Default (Testing)"):
            st.markdown("*⚠️ Ganti sebelum production!*")


# ── HOME / NEWS PAGE ──────────────────────────────────────────────────────────
def home_page():
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,255,255,0.15) 0%,rgba(138,43,226,0.15) 100%);
                border:2px solid rgba(0,255,255,0.3);border-radius:20px;
                padding:2rem 2.5rem;margin-bottom:2rem;text-align:center;">
        <h1 style="font-family:Orbitron,sans-serif;color:#00ffff;font-size:2rem;font-weight:900;
                   letter-spacing:3px;margin:0 0 0.5rem 0;text-shadow:0 0 20px rgba(0,255,255,0.6);">
            📡 MEDICAL NEWS CENTER
        </h1>
        <p style="font-family:Rajdhani,sans-serif;color:#a0a0ff;font-size:1rem;letter-spacing:1px;margin:0;">
            Berita &amp; riset terkini seputar tumor otak dan neuro-onkologi
        </p>
    </div>
    """, unsafe_allow_html=True)

    cq, cs, cb = st.columns([3, 2, 1])
    with cq: selected_query = st.selectbox("Topik", NEWS_QUERIES, label_visibility="collapsed")
    with cs: source_mode = st.selectbox("Sumber", ["GNews.io (Live)", "ScienceDaily RSS (Research)", "Keduanya"], index=2, label_visibility="collapsed")
    with cb:
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown(f"""
    <div style="background:rgba(0,255,255,0.05);border:1px solid rgba(0,255,255,0.15);
                border-radius:8px;padding:8px 16px;margin-bottom:1.5rem;
                font-family:Rajdhani,sans-serif;font-size:0.85rem;color:#6060a0;">
        ⏱ Cache 1 jam &nbsp;|&nbsp; Dimuat: {datetime.now().strftime('%d %b %Y, %H:%M')} &nbsp;|&nbsp; GNews + ScienceDaily
    </div>
    """, unsafe_allow_html=True)

    show_g = source_mode in ["GNews.io (Live)", "Keduanya"]
    show_r = source_mode in ["ScienceDaily RSS (Research)", "Keduanya"]

    if source_mode == "Keduanya":
        cl, cr = st.columns(2, gap="medium")
    else:
        cl = cr = st.container()

    if show_g:
        with cl:
            st.markdown("""<h3 style="font-family:Orbitron,sans-serif;color:#00ffff;font-size:1rem;
                font-weight:700;letter-spacing:2px;margin-bottom:1rem;">🌐 BERITA TERKINI (GNews)</h3>""",
                unsafe_allow_html=True)
            with st.spinner("Memuat berita..."):
                arts = fetch_gnews(selected_query, max_results=6 if source_mode == "Keduanya" else 8)
            if arts:
                for a in arts: render_news_card(a)
            else:
                st.warning("⚠️ Tidak ada berita dari GNews. Limit 100 req/hari — coba ganti topik.")

    if show_r:
        with cr:
            st.markdown("""<h3 style="font-family:Orbitron,sans-serif;color:#8a2be2;font-size:1rem;
                font-weight:700;letter-spacing:2px;margin-bottom:1rem;">🔬 RISET ILMIAH (ScienceDaily)</h3>""",
                unsafe_allow_html=True)
            all_rss = []
            for name, url in RSS_FEEDS.items():
                with st.spinner(f"Memuat {name}..."):
                    all_rss.extend(fetch_rss(name, url, max_results=4))
            if all_rss:
                for a in all_rss: render_news_card(a)
            else:
                st.warning("⚠️ Gagal memuat RSS.")

    st.markdown("""
    <div style="text-align:center;padding:1rem;border-top:1px solid rgba(0,255,255,0.15);
                color:#6060a0;font-family:Rajdhani,sans-serif;font-size:0.8rem;margin-top:2rem;">
        Sumber: <a href="https://gnews.io" target="_blank" style="color:#00ffff;">GNews.io</a> &amp;
        <a href="https://www.sciencedaily.com/news/health_medicine/brain_tumor/" target="_blank" style="color:#8a2be2;">ScienceDaily</a>
        — konten untuk keperluan informatif dan edukasi.
    </div>
    """, unsafe_allow_html=True)


# ── KLASIFIKASI MRI ───────────────────────────────────────────────────────────
def main_app():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🧠 NEUROSCAN AI</h1>
            <p class="header-subtitle">Advanced Brain Tumor Detection System | GLCM + Random Forest</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""<div style="text-align:center;margin-top:1rem;">
            <div class="welcome-badge">👤 {st.session_state.username.upper()}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ SYSTEM SETTINGS")
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        sav_files = []
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.venv']]
            for f in files:
                if f.endswith(('.sav', '.pkl')):
                    sav_files.append(os.path.relpath(os.path.join(root, f), base_dir))

        if sav_files:
            default_idx = next((i for i, p in enumerate(sav_files)
                                if 'rf' in p.lower() and 'best' not in p.lower()), 0)
            model_path = st.selectbox("Pilih Model", sav_files, index=default_idx,
                                      help="File .sav yang terdeteksi di repository")
        else:
            st.warning("⚠️ Tidak ada file .sav ditemukan.")
            model_path = st.text_input("Model Path (manual)", value="brain_tumor_rf_glcm_model.sav")

        st.markdown("---")
        st.markdown("### 📊 MODEL INFORMATION")
        st.info("""
**Model:** Random Forest Classifier

**Features:** GLCM
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
        st.markdown("1. Upload gambar MRI otak\n2. Klik **START AI ANALYSIS**\n3. Lihat hasil diagnosis\n4. Download laporan CSV")

    resolved = resolve_model_path(model_path)
    if resolved is None:
        st.error(f"❌ Model tidak ditemukan: `{model_path}`")
        st.info("💡 Pastikan file .sav ada di repo GitHub.")
        return

    model = load_model(model_path)
    if model is None:
        st.error("❌ Gagal load model.")
        return

    with st.sidebar:
        st.success(f"✅ Model aktif:\n`{os.path.basename(resolved)}`")

    st.markdown("### 📤 UPLOAD BRAIN MRI SCANS")
    uploaded_files = st.file_uploader(
        "Pilih gambar MRI (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
        st.info("👆 **UPLOAD MRI SCANS UNTUK MEMULAI ANALISIS**")
        st.markdown("""
### 📌 KEMAMPUAN SISTEM
- **🔴 Glioma Tumor** — Tumor ganas dari sel glial otak
- **🔵 Meningioma Tumor** — Tumor dari meninges
- **🟢 No Tumor** — Jaringan otak normal
- **🟡 Pituitary Tumor** — Tumor pada kelenjar hipofisis

**Format:** PNG, JPG, JPEG
        """)
        return

    st.success(f"✅ **{len(uploaded_files)} gambar berhasil diupload**")
    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
    st.markdown("### 🖼️ PREVIEW SCAN")

    cols = st.columns(min(len(uploaded_files), 4))
    for idx, uf in enumerate(uploaded_files):
        with cols[idx % 4]:
            st.image(Image.open(uf), caption=uf.name, use_container_width=True)

    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
    _, c, _ = st.columns([1, 2, 1])
    with c:
        predict_button = st.button("🔍 START AI ANALYSIS", use_container_width=True)

    if not predict_button:
        return

    progress = st.progress(0)
    status   = st.empty()
    results  = []

    for idx, uf in enumerate(uploaded_files):
        status.text(f"⚡ Menganalisis {uf.name}...")
        image = Image.open(uf)
        pc, conf, prob = predict_image(image, model)
        if pc is not None:
            results.append({'filename': uf.name, 'image': image,
                            'predicted_class': pc, 'confidence': conf, 'probabilities': prob})
        progress.progress((idx + 1) / len(uploaded_files))

    status.text("✅ Analisis selesai!")
    progress.empty()

    if not results:
        st.error("❌ Tidak ada gambar yang berhasil diproses.")
        return

    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)
    st.markdown("## 🎯 HASIL ANALISIS")

    class_counts = {}
    for r in results: class_counts[r['predicted_class']] = class_counts.get(r['predicted_class'], 0) + 1

    c1, c2, c3, c4 = st.columns(4)
    for col, key, color, label in [
        (c1, 'Glioma Tumor',     '#ff006e', 'GLIOMA'),
        (c2, 'Meningioma Tumor', '#00b4d8', 'MENINGIOMA'),
        (c3, 'No Tumor',         '#06ffa5', 'NO TUMOR'),
        (c4, 'Pituitary Tumor',  '#ffd60a', 'PITUITARY'),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="margin:0;font-size:2.5rem;color:{color};">{class_counts.get(key, 0)}</h3>
                <p style="margin:0;font-family:Rajdhani;letter-spacing:1px;">{label}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)

    for idx, r in enumerate(results):
        st.markdown(f"### 📋 SCAN RESULT #{idx + 1}")
        ca, cb_ = st.columns([1, 2])
        with ca:
            st.image(r['image'], use_container_width=True)
        with cb_:
            st.markdown(f"**📁 FILE:** `{r['filename']}`")
            badge = get_badge_class(r['predicted_class'])
            st.markdown(f"""
            <div>
                <strong style="color:#00ffff;">🎯 DIAGNOSIS:</strong>
                <span class="result-badge {badge}">{r['predicted_class'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)
            pct = r['confidence'] * 100
            st.markdown(f"**📊 CONFIDENCE:** `{pct:.2f}%`")
            st.progress(r['confidence'])
            st.plotly_chart(create_probability_chart(r['probabilities'],
                            f"Probabilitas — {r['filename']}"), use_container_width=True)
        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)

    st.markdown("### 💾 DOWNLOAD LAPORAN")
    df_out = pd.DataFrame([{
        'Filename':            r['filename'],
        'Predicted Class':     r['predicted_class'],
        'Confidence (%)':      f"{r['confidence']*100:.2f}",
        'Glioma Prob (%)':     f"{r['probabilities']['Glioma Tumor']*100:.2f}",
        'Meningioma Prob (%)': f"{r['probabilities']['Meningioma Tumor']*100:.2f}",
        'No Tumor Prob (%)':   f"{r['probabilities']['No Tumor']*100:.2f}",
        'Pituitary Prob (%)':  f"{r['probabilities']['Pituitary Tumor']*100:.2f}",
    } for r in results])
    _, cd, _ = st.columns([1, 2, 1])
    with cd:
        st.download_button(
            label="📥 DOWNLOAD LAPORAN (CSV)",
            data=df_out.to_csv(index=False),
            file_name=f"neuroscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", use_container_width=True
        )


# ── DEVELOPER PAGE ────────────────────────────────────────────────────────────
def developer_page():
    _, c1, _ = st.columns([1, 2, 1])
    with c1:
        if st.button("⬅️ KEMBALI", use_container_width=True):
            st.session_state.show_developer_page = False
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown("""
        <div class="dev-profile-container">
            <div class="dev-profile-title">👩‍💻 DEVELOPER PROFILE</div>
            <div class="tech-line"></div>
        </div>
        """, unsafe_allow_html=True)

        base_dir   = os.path.dirname(os.path.abspath(__file__))
        photo_path = next((os.path.join(base_dir, f)
                           for f in ['profil.jpeg', 'profil.jpg', 'profil.png']
                           if os.path.exists(os.path.join(base_dir, f))), None)
        if photo_path:
            try:
                _, pb, _ = st.columns([1, 1, 1])
                with pb: st.image(Image.open(photo_path), use_container_width=True)
            except Exception:
                pass
        else:
            st.markdown("""
            <div style="text-align:center;margin:1.5rem 0;">
                <div style="width:180px;height:180px;margin:0 auto;
                            background:linear-gradient(135deg,#00ffff 0%,#8a2be2 100%);
                            border-radius:50%;display:flex;align-items:center;
                            justify-content:center;font-size:4rem;border:5px solid #00ffff;
                            box-shadow:0 0 30px rgba(0,255,255,0.6);">👩‍💻</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='tech-line'></div>", unsafe_allow_html=True)

        for label, value in [
            ("👤 FULL NAME",            "JIHAN SUCI ANANDA"),
            ("📍 PLACE & DATE OF BIRTH", "Sungai Penuh, 25 Januari 2002"),
            ("💼 POSITION",             "AI/ML Engineer & Data Scientist"),
        ]:
            st.markdown(f"""
            <div class="dev-info-section">
                <div class="dev-info-label">{label}</div>
                <div class="dev-info-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="dev-info-section">
            <div class="dev-info-label">📝 ABOUT ME</div>
            <div class="dev-bio">
                Saya adalah seorang AI/ML Engineer yang passionate dalam mengembangkan solusi berbasis
                kecerdasan buatan untuk menyelesaikan masalah kompleks di bidang kesehatan dan teknologi.
                Dengan pengalaman dalam pengembangan sistem deteksi tumor otak menggunakan machine learning,
                saya berkomitmen untuk menciptakan aplikasi yang dapat membantu tenaga medis dalam proses
                diagnosis yang lebih akurat dan efisien. Keahlian saya mencakup Computer Vision, Deep Learning,
                dan pengembangan aplikasi berbasis web yang user-friendly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        skills = ["Python", "Machine Learning", "Computer Vision", "Scikit-learn",
                  "Streamlit", "OpenCV", "Data Analysis", "GLCM"]
        skill_pills = "".join([
            f'<span style="display:inline-block;background:linear-gradient(135deg,#00ffff 0%,#8a2be2 100%);'
            f'padding:0.4rem 0.9rem;border-radius:15px;margin:0.25rem;color:white;font-family:Rajdhani;">{s}</span>'
            for s in skills])
        st.markdown(f"""
        <div class="dev-info-section">
            <div class="dev-info-label">🛠️ TECHNICAL SKILLS</div>
            <div style="margin-top:1rem;">{skill_pills}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center;color:#a0a0ff;font-family:Rajdhani;margin-top:2rem;">
            <p>⚡ NeuroScan AI Version 3.0 — {datetime.now().year}</p>
            <p style="font-size:0.9rem;opacity:0.7;">Powered by Machine Learning & Computer Vision</p>
        </div>
        """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                      MAIN ROUTER                         ║
# ╚══════════════════════════════════════════════════════════╝

def main():
    if not st.session_state.logged_in:
        login_page()
        return

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:1rem;">
            <div class="welcome-badge">👤 {st.session_state.username.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### 🧭 NAVIGASI")

        if st.button("🏠  Beranda & Berita", use_container_width=True,
                     type="primary" if st.session_state.current_page == "home" and not st.session_state.show_developer_page else "secondary"):
            st.session_state.current_page        = "home"
            st.session_state.show_developer_page = False
            st.rerun()

        if st.button("🧠  Klasifikasi MRI", use_container_width=True,
                     type="primary" if st.session_state.current_page == "predict" and not st.session_state.show_developer_page else "secondary"):
            st.session_state.current_page        = "predict"
            st.session_state.show_developer_page = False
            st.rerun()

        if st.button("👩‍💻  Developer", use_container_width=True,
                     type="primary" if st.session_state.show_developer_page else "secondary"):
            st.session_state.show_developer_page = True
            st.rerun()

        st.markdown("---")

        if st.button("🚪 LOGOUT", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown(f"<p style='color:#6060a0;font-size:0.8rem;text-align:center;margin-top:1rem;'>NeuroScan AI v3.0 — {datetime.now().year}</p>",
                    unsafe_allow_html=True)

    if st.session_state.show_developer_page:
        developer_page()
    elif st.session_state.current_page == "home":
        home_page()
    else:
        main_app()


if __name__ == "__main__":
    main()
