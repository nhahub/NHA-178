"""
Home Dashboard Page
"""

import streamlit as st
import os

def show():
    # Hero section with consistent styling
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>🏥</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                Diabetes Classification Dashboard
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Production-Ready ML Pipeline with MLflow Integration
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Project overview
    st.markdown("## 📋 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 What This Application Does
        
        This interactive dashboard provides a complete machine learning solution for predicting diabetes 
        using the Pima Indians Diabetes dataset. Built with industry best practices and MLflow integration.
        
        **Key Features:**
        - 🤖 8 ML algorithms with 2-stage optimization pipeline
        - ⚡ Stage 1: Quick baseline evaluation of all models
        - 🎯 Stage 2: GridSearchCV on top performers
        - 🔮 Interactive prediction interface with feature engineering
        - 📊 Comprehensive EDA and XAI analysis
        - 📁 MLflow experiment tracking and model registry
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Model Performance
        
        Our optimized ensemble model achieves:
        - **Accuracy:** 88.96%
        - **ROC-AUC:** 91.89%
        - **F1-Score:** 84.62%
        - **Precision:** 85.19%
        - **Recall:** 84.06%
        
        Trained on 768 instances with 24 engineered features using 2-stage optimization.
        """)
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("## 📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>9</div>
                <div class='metric-label'>ML Algorithms</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>24</div>
                <div class='metric-label'>Features</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>88.96%</div>
                <div class='metric-label'>Best Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class='metric-card'>
                <div class='metric-value'>91.89%</div>
                <div class='metric-label'>Best ROC-AUC</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action cards
    st.markdown("## 🚀 Get Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                <h3 style='color: #0066cc;'>📊 Explore Data</h3>
                <p style='color: #666;'>Upload and visualize your dataset with comprehensive EDA tools</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                <h3 style='color: #0066cc;'>🔧 Train Models</h3>
                <p style='color: #666;'>Train new models with custom hyperparameters or use defaults</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 30px; border-radius: 15px; 
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;'>
                <h3 style='color: #0066cc;'>🔮 Make Predictions</h3>
                <p style='color: #666;'>Get instant predictions with probability scores and explanations</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛠️ Technologies Used")
        st.markdown("""
        - **Python** 3.8+
        - **Streamlit** - Interactive web framework
        - **MLflow** - Experiment tracking
        - **Scikit-learn** - ML algorithms
        - **XGBoost & LightGBM** - Gradient boosting
        - **Optuna** - Hyperparameter optimization
        """)
    
    with col2:
        st.markdown("### 📚 Available Models")
        st.markdown("""
        1. Logistic Regression
        2. Random Forest
        3. XGBoost
        4. LightGBM
        5. SVM
        6. Gradient Boosting
        7. KNN
        8. Decision Tree
        9. Ensemble (Best Performance)
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; 
                    border-radius: 10px; margin-top: 2rem;'>
            <p style='color: #666; margin: 0;'>
                Made with ❤️ by <strong>Hossam Medhat</strong> | 
                <a href='mailto:hossammedhat81@gmail.com'>hossammedhat81@gmail.com</a>
            </p>
            <p style='color: #999; margin-top: 0.5rem; font-size: 0.9em;'>
                Last Updated: November 25, 2025
            </p>
        </div>
    """, unsafe_allow_html=True)
