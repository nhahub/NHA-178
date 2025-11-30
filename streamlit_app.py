"""
Pima Indians Diabetes Classification - Streamlit Web Application
Production-ready ML Dashboard with MLflow Integration
Author: Quattro Xpert
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Diabetes ML Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme styling
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main App Background */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(138, 43, 226, 0.3);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .st-emotion-cache-1d391kg {
        color: #e0e0e0;
    }
    
    /* Headers with Gradient */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
    }
    
    h2 {
        color: #a78bfa;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #c4b5fd;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton>button:active {
        transform: translateY(0px);
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid rgba(167, 139, 250, 0.3);
        font-size: 2rem;
        font-weight: 700;
        color: #a78bfa !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
        font-weight: 500;
    }
    
    /* Custom Metric Card */
    .metric-card {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.2);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        border-color: rgba(167, 139, 250, 0.5);
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1em;
        color: #a78bfa;
        margin-top: 5px;
        font-weight: 500;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background-color: rgba(26, 26, 46, 0.6);
        color: #e0e0e0;
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 10px;
        padding: 0.5rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sliders */
    .stSlider>div>div>div>div {
        background-color: #667eea;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(26, 26, 46, 0.6);
        border: 2px dashed rgba(167, 139, 250, 0.4);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* DataFrames */
    .dataframe {
        background-color: rgba(26, 26, 46, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(167, 139, 250, 0.2);
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        background-color: rgba(26, 26, 46, 0.8);
        border-radius: 12px;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 10px;
        color: #a78bfa;
        font-weight: 500;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(26, 26, 46, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a78bfa;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Links */
    a {
        color: #a78bfa;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    a:hover {
        color: #667eea;
    }
    
    /* Custom Cards */
    .glass-card {
        background: rgba(26, 26, 46, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(167, 139, 250, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #a78bfa;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        color: #a78bfa;
    }
    
    /* Text */
    p, li, label, .css-10trblm {
        color: #d1d5db;
    }
    
    /* Divider */
    hr {
        border-color: rgba(167, 139, 250, 0.2);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(26, 26, 46, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem;
        text-align: center;
        border-top: 1px solid rgba(167, 139, 250, 0.3);
        z-index: 999;
    }
    
    .footer-text {
        color: #a78bfa;
        font-weight: 500;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'active_model_path' not in st.session_state:
    st.session_state.active_model_path = None
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = None

# Sidebar navigation with visible title and emojis
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 0.5rem; background: rgba(102, 126, 234, 0.05); border-radius: 12px;'>
        <div style='font-size: 3.5rem; margin-bottom: 0.8rem; filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.6));'>🏥</div>
        <h2 style='font-size: 1.5rem; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
            Diabetes ML Dashboard
        </h2>
    </div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "🧭 Navigation",
    ["🏠 Home", "📊 Dataset Explorer", "🔧 Train Model", "⚡ Advanced Training", "🔮 Make Predictions", "🔬 XAI Analysis", "📁 MLflow Models"],
    index=0
)

st.sidebar.markdown("---")

# Sidebar info with modern styling and visible emojis
st.sidebar.markdown("""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(167, 139, 250, 0.3);'>
        <p style='margin: 0; color: #e0e0e0; font-weight: 600; font-size: 1.1rem;'>
            <span style='font-size: 1.3rem; filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5));'>👨‍💻</span> Author
        </p>
        <p style='margin: 0.5rem 0; color: #ffffff; font-size: 1.05rem; font-weight: 500;'>Quattro Xpert</p>
        <p style='margin: 1.2rem 0 0 0; color: #e0e0e0; font-weight: 600; font-size: 1.1rem;'>
            <span style='font-size: 1.3rem; filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5));'>📋</span> Project
        </p>
        <p style='margin: 0.5rem 0 0 0; color: #ffffff; font-size: 0.95rem; line-height: 1.4;'>Pima Indians Diabetes Classification</p>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Import page modules
from app.pages import home, dataset_explorer, training, predict, model_explorer
try:
    from app.pages import training_enhanced, xai_analysis
    has_advanced = True
except ImportError:
    has_advanced = False

# Route to pages
if page == "🏠 Home":
    home.show()
elif page == "📊 Dataset Explorer":
    dataset_explorer.show()
elif page == "🔧 Train Model":
    training.show()
elif page == "⚡ Advanced Training":
    if has_advanced:
        training_enhanced.show()
    else:
        st.error("Advanced training module not available")
elif page == "🔮 Make Predictions":
    predict.show()
elif page == "🔬 XAI Analysis":
    if has_advanced:
        xai_analysis.show()
    else:
        st.error("XAI analysis module not available")
elif page == "📁 MLflow Models":
    model_explorer.show()

# Footer
st.markdown("""
    <div class='footer'>
        <p class='footer-text'>Made with ❤️ by <strong>Quattro Xpert</strong> | Pima Indians Diabetes ML Dashboard</p>
    </div>
    <div style='height: 60px;'></div>
""", unsafe_allow_html=True)
