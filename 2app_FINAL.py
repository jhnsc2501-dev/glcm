"""
Brain Tumor Classification App - PRODUCTION READY
Using GLCM Features + Random Forest Classifier

Author: Pertamina EP Jambi Field
Date: 2025
Version: 1.0 FINAL
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

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Result badge */
    .result-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .badge-glioma {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-meningioma {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .badge-no-tumor {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    .badge-pituitary {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================

@st.cache_resource
def load_model(model_path):
    """Load the trained Random Forest model - supports both pickle and joblib"""
    try:
        # Try pickle first
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Model loaded successfully with pickle!")
        return model
    except Exception as pickle_error:
        try:
            # If pickle fails, try joblib
            import joblib
            model = joblib.load(model_path)
            st.success("✅ Model loaded successfully with joblib!")
            return model
        except Exception as joblib_error:
            st.error(f"❌ Error loading model with pickle: {str(pickle_error)}")
            st.error(f"❌ Error loading model with joblib: {str(joblib_error)}")
            return None


def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Extract GLCM features from an image
    
    Parameters:
    - image: numpy array (grayscale image)
    - distances: list of pixel pair distances
    - angles: list of angles in radians
    
    Returns:
    - Dictionary containing GLCM features
    """
    try:
        # Resize image to standard size
        img = cv2.resize(image, (128, 128))
        
        # Normalize to 0-255 range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(img, distances=distances, angles=angles, 
                            levels=256, symmetric=True, normed=True)
        
        # Extract GLCM properties
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
    """
    Predict the class of a brain tumor image
    
    Parameters:
    - image: PIL Image or numpy array
    - model: trained Random Forest model
    
    Returns:
    - predicted_class: string
    - confidence: float
    - probabilities: dict
    """
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Extract GLCM features
        features = extract_glcm_features(image)
        
        if features is None:
            return None, None, None
        
        # Convert to array and reshape
        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Predict
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]
        
        # Class mapping
        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        
        predicted_class = class_names[prediction]
        confidence = probabilities[prediction]
        
        # Create probability dictionary
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
    
    # Sort by probability
    df = df.sort_values('Probability', ascending=True)
    
    # Create color map
    colors = []
    for class_name in df['Class']:
        if 'glioma' in class_name.lower():
            colors.append('#f5576c')
        elif 'meningioma' in class_name.lower():
            colors.append('#00f2fe')
        elif 'no tumor' in class_name.lower():
            colors.append('#43e97b')
        else:
            colors.append('#fee140')
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['Probability'],
            y=df['Class'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=2)
            ),
            text=[f'{p:.1f}%' for p in df['Probability']],
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Probability (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    fig.update_xaxes(range=[0, 100], gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
    
    return fig


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🧠 Brain Tumor Classifier</h1>
        <p class="header-subtitle">AI-Powered Brain Tumor Detection using GLCM Features & Random Forest</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model file path
        model_path = st.text_input(
            "Model Path",
            value="models/brain_tumor_rf_glcm_model.pkl",
            help="Path to your trained Random Forest model (.pkl or .sav file)"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Information")
        st.info("""
        **Model Type:** Random Forest Classifier
        
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
        
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Upload one or more brain MRI images (PNG, JPG, JPEG)
        2. Click **🔍 Predict** button
        3. View results with confidence scores
        4. Analyze probability distribution
        """)
        
        st.markdown("---")
        
        st.markdown("### 👨‍💻 Developer")
        st.markdown("**Pertamina EP Jambi Field**")
        st.markdown(f"*Version 1.0 - {datetime.now().year}*")
    
    # Load model
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is None:
            st.error("❌ Failed to load model. Please check the model file.")
            st.info("💡 Make sure the model file is a valid pickle (.pkl) or joblib (.sav) file.")
            return
    else:
        st.error(f"❌ Model file not found: {model_path}")
        st.info("💡 Please update the model path in the sidebar settings.")
        st.info("📝 Default path: models/brain_tumor_rf_glcm_model.pkl")
        return
    
    # File uploader
    st.markdown("### 📤 Upload Brain MRI Images")
    uploaded_files = st.file_uploader(
        "Choose brain MRI images (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="You can upload multiple images at once"
    )
    
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        
        # Display uploaded images in a grid
        st.markdown("---")
        st.markdown("### 🖼️ Uploaded Images Preview")
        
        cols = st.columns(min(len(uploaded_files), 4))
        for idx, uploaded_file in enumerate(uploaded_files):
            with cols[idx % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        st.markdown("---")
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🔍 Predict All Images", use_container_width=True)
        
        if predict_button:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Load image
                image = Image.open(uploaded_file)
                
                # Predict
                predicted_class, confidence, probabilities = predict_image(image, model)
                
                if predicted_class is not None:
                    results.append({
                        'filename': uploaded_file.name,
                        'image': image,
                        'predicted_class': predicted_class,
                        'confidence': confidence,
                        'probabilities': probabilities
                    })
                else:
                    st.warning(f"⚠️ Failed to process {uploaded_file.name}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Prediction completed!")
            progress_bar.empty()
            
            if len(results) == 0:
                st.error("❌ No images were successfully processed.")
                return
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Count predictions by class
            class_counts = {}
            for result in results:
                cls = result['predicted_class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            with col1:
                glioma_count = class_counts.get('Glioma Tumor', 0)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3 style="margin:0; font-size:2rem;">{glioma_count}</h3>
                    <p style="margin:0;">Glioma</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                meningioma_count = class_counts.get('Meningioma Tumor', 0)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3 style="margin:0; font-size:2rem;">{meningioma_count}</h3>
                    <p style="margin:0;">Meningioma</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                no_tumor_count = class_counts.get('No Tumor', 0)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h3 style="margin:0; font-size:2rem;">{no_tumor_count}</h3>
                    <p style="margin:0;">No Tumor</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                pituitary_count = class_counts.get('Pituitary Tumor', 0)
                st.markdown(f"""
                <div class="metric-container" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                    <h3 style="margin:0; font-size:2rem;">{pituitary_count}</h3>
                    <p style="margin:0;">Pituitary</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed results for each image
            for idx, result in enumerate(results):
                st.markdown(f"### 📋 Result #{idx + 1}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(result['image'], use_container_width=True)
                
                with col2:
                    # File name
                    st.markdown(f"**📁 Filename:** `{result['filename']}`")
                    
                    # Prediction with badge
                    badge_class = get_badge_class(result['predicted_class'])
                    st.markdown(f"""
                    <div>
                        <strong>🎯 Prediction:</strong>
                        <span class="result-badge {badge_class}">{result['predicted_class']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence
                    confidence_percent = result['confidence'] * 100
                    st.markdown(f"**📊 Confidence:** `{confidence_percent:.2f}%`")
                    
                    # Progress bar for confidence
                    st.progress(result['confidence'])
                    
                    # Probability chart
                    fig = create_probability_chart(
                        result['probabilities'],
                        title=f"Class Probabilities - {result['filename']}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
            
            # Download results as CSV
            st.markdown("### 💾 Download Results")
            
            # Create DataFrame
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
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"brain_tumor_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        # Empty state
        st.markdown("---")
        st.info("👆 Please upload brain MRI images to start prediction")
        
        # Show example
        st.markdown("### 📌 Example")
        st.markdown("""
        Upload brain MRI images in the following formats:
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        
        The model will analyze the images and predict:
        - **Glioma Tumor** - A type of brain tumor that starts in glial cells
        - **Meningioma Tumor** - A tumor that forms in membranes covering the brain
        - **No Tumor** - Healthy brain scan with no tumor detected
        - **Pituitary Tumor** - A tumor in the pituitary gland
        """)


# ==================== RUN APP ====================
if __name__ == "__main__":
    main()
