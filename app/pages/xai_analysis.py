"""
Explainable AI (XAI) Analysis Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def show():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>🔬</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                Explainable AI Analysis
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Understand model decisions through feature importance and SHAP analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model selection
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>📂 Step 1: Load Model for Analysis</h2>
            <p style='color: #a78bfa;'>Select a trained model to analyze</p>
        </div>
    """, unsafe_allow_html=True)
    
    model = None
    model_name = "Unknown"
    
    # Check for active model
    if 'active_model_path' in st.session_state and st.session_state.active_model_path:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📊 **Active Model:** {st.session_state.get('active_model_name', 'Trained Model')}")
        with col2:
            if st.button("✅ Use This Model", use_container_width=True, type="primary"):
                try:
                    model = joblib.load(st.session_state.active_model_path)
                    model_name = st.session_state.get('active_model_name', 'Trained Model')
                    st.success(f"✅ Model loaded: {model_name}")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
    
    # Upload model option
    st.markdown("**Or upload a model file:**")
    uploaded_model = st.file_uploader(
        "Upload .pkl model file",
        type=['pkl'],
        key="xai_model_upload"
    )
    
    if uploaded_model is not None and model is None:
        try:
            model = joblib.load(uploaded_model)
            model_name = uploaded_model.name
            st.success(f"✅ Model loaded: {model_name}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    if model is None:
        st.warning("⚠️ Please load a model to perform XAI analysis")
        return
    
    st.markdown("---")
    
    # Check model capabilities
    has_feature_importance = hasattr(model, 'feature_importances_')
    has_predict_proba = hasattr(model, 'predict_proba')
    
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>🎯 Model Capabilities</h2>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅" if has_feature_importance else "❌"
        st.metric("Feature Importance", status)
    with col2:
        status = "✅" if has_predict_proba else "❌"
        st.metric("Probability Predictions", status)
    with col3:
        model_type = type(model).__name__
        st.metric("Model Type", model_type)
    
    st.markdown("---")
    
    # Feature Importance Analysis
    if has_feature_importance:
        st.markdown("""
            <div class='glass-card'>
                <h2 style='margin-top: 0;'>📊 Feature Importance Analysis</h2>
                <p style='color: #a78bfa;'>Discover which features contribute most to predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Create feature names (assuming standard preprocessing)
            feature_names = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                'N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 
                'N8', 'N9', 'N10', 'N11', 'N12', 'N13', 'N14', 'N15'
            ]
            
            # Handle ensemble models
            if hasattr(model, 'estimators_'):
                # For ensemble, average importance across base models
                if len(importances) == len(feature_names):
                    pass  # Already correct
                else:
                    st.warning("⚠️ Feature count mismatch. Using available features.")
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
            
            # Ensure matching lengths
            if len(importances) != len(feature_names):
                feature_names = feature_names[:len(importances)]
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Top features selection
            n_features = st.slider("Number of top features to display", 5, 24, 15, 1)
            
            top_features = importance_df.head(n_features)
            
            # Display table
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Plot
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 8))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features['Feature'])
                ax.set_xlabel('Importance', color='#d1d5db', fontsize=12, fontweight='bold')
                ax.set_title(f'Top {n_features} Feature Importances', color='#a78bfa', fontsize=14, fontweight='bold', pad=15)
                ax.tick_params(colors='#d1d5db')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.2, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#a78bfa')
                ax.spines['bottom'].set_color('#a78bfa')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
                    ax.text(val, i, f' {val:.4f}', va='center', color='white', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Top Features")
                st.dataframe(
                    top_features.reset_index(drop=True).style.background_gradient(cmap='viridis', subset=['Importance']),
                    use_container_width=True,
                    height=400
                )
            
            st.markdown("---")
            
            # Feature categories
            st.markdown("#### 📁 Feature Categories")
            
            original_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            engineered_features = [f'N{i}' for i in range(16)]
            
            # Categorize importances
            original_importance = importance_df[importance_df['Feature'].isin(original_features)]['Importance'].sum()
            engineered_importance = importance_df[importance_df['Feature'].isin(engineered_features)]['Importance'].sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Features", f"{original_importance:.4f}", 
                         help="Combined importance of 8 original features")
            
            with col2:
                st.metric("Engineered Features", f"{engineered_importance:.4f}",
                         help="Combined importance of 16 engineered features")
            
            with col3:
                ratio = (engineered_importance / original_importance) if original_importance > 0 else 0
                st.metric("Engineered/Original Ratio", f"{ratio:.2f}x")
            
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            colors_pie = ['#667eea', '#764ba2']
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(
                [original_importance, engineered_importance],
                labels=['Original Features', 'Engineered Features'],
                autopct='%1.1f%%',
                colors=colors_pie,
                explode=explode,
                textprops={'color': 'white', 'fontweight': 'bold', 'fontsize': 12},
                startangle=90
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(14)
            
            ax.set_title('Feature Category Importance', color='#a78bfa', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            st.warning("⚠️ This model type doesn't support feature importance")
    
    else:
        st.info("ℹ️ Feature importance not available for this model type (e.g., SVM, KNN)")
        st.markdown("""
            **Supported models:**
            - Tree-based: Random Forest, XGBoost, LightGBM, Gradient Boosting, Decision Tree
            - Ensemble models with tree-based estimators
            
            **Not supported:**
            - Linear models (Logistic Regression, SVM)
            - Distance-based (KNN)
        """)
    
    st.markdown("---")
    
    # SHAP Analysis (placeholder - requires SHAP library)
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>🎯 SHAP Analysis</h2>
            <p style='color: #a78bfa;'>Coming soon: Interactive SHAP visualizations</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("""
        📚 **SHAP (SHapley Additive exPlanations)** provides:
        - Individual prediction explanations
        - Feature contribution waterfall plots
        - Summary plots across all predictions
        - Dependency plots for feature interactions
        
        Install SHAP library to enable: `pip install shap`
    """)
    
    st.markdown("---")
    
    # Model interpretation tips
    with st.expander("💡 How to Interpret Feature Importance"):
        st.markdown("""
        ### Understanding Feature Importance
        
        **What is Feature Importance?**
        - Measures how much each feature contributes to the model's predictions
        - Higher values = more important features
        - Sum of all importances = 1.0 (100%)
        
        **How to Use This Information:**
        1. **Focus on Top Features**: The most important features have the greatest impact on predictions
        2. **Feature Engineering Validation**: Check if engineered features (N0-N15) are adding value
        3. **Data Collection**: Prioritize collecting accurate data for high-importance features
        4. **Model Simplification**: Consider removing very low-importance features
        
        **Original Features:**
        - `Glucose`: Blood glucose level (mg/dL)
        - `BMI`: Body Mass Index
        - `Age`: Patient age (years)
        - `DiabetesPedigreeFunction`: Genetic factor
        - `Insulin`: Insulin level (μU/mL)
        - `Pregnancies`: Number of pregnancies
        - `BloodPressure`: Blood pressure (mm Hg)
        - `SkinThickness`: Skin fold thickness (mm)
        
        **Engineered Features (N0-N15):**
        - Binary indicators and interaction terms
        - Capture complex relationships between original features
        - Example: `N0 = BMI × SkinThickness`
        """)

