"""
Model Training Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.utils.model_utils import (
    get_model, preprocess_data, train_model_with_mlflow,
    create_ensemble_model
)
from app.utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

def show():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>🔧</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                Model Training
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Train and optimize machine learning models with custom hyperparameters
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Upload dataset
    st.subheader("📂 Step 1: Upload Training Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV File", 
        type=['csv'],
        help="Upload your preprocessed diabetes dataset",
        key="training_upload"
    )
    
    col1, col2 = st.columns([3, 1])
    with col2:
        use_default = st.button("📂 Use Default Dataset", use_container_width=True, key="default_train")
    
    if uploaded_file is None and not use_default:
        st.info("👆 Please upload a dataset to begin training")
        return
    
    # Load data
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            # Load default dataset
            import kagglehub
            path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
            csv_path = os.path.join(path, "diabetes.csv")
            data = pd.read_csv(csv_path)
        
        st.success(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
        # Verify required columns
        if 'Outcome' not in data.columns:
            st.error("❌ Dataset must contain 'Outcome' column")
            return
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return
    
    st.markdown("---")
    
    # Model selection
    st.subheader("🤖 Step 2: Select Model Type")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Choose Model",
            [
                "Random Forest",
                "XGBoost", 
                "LightGBM",
                "Logistic Regression",
                "SVM",
                "Gradient Boosting",
                "KNN",
                "Decision Tree",
                "Ensemble (Best Performance)"
            ],
            help="Select the machine learning algorithm to train"
        )
    
    with col2:
        use_custom_params = st.checkbox(
            "Use Custom Hyperparameters",
            value=False,
            help="Enable to customize model hyperparameters"
        )
    
    # Hyperparameter configuration
    custom_params = {}
    
    if use_custom_params:
        st.markdown("#### ⚙️ Hyperparameter Configuration")
        
        with st.expander("Configure Hyperparameters", expanded=True):
            if model_type == "Random Forest":
                col1, col2, col3 = st.columns(3)
                with col1:
                    custom_params['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 50)
                with col2:
                    custom_params['max_depth'] = st.slider("Max Depth", 3, 30, 10, 1)
                with col3:
                    custom_params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2, 1)
            
            elif model_type == "XGBoost":
                col1, col2, col3 = st.columns(3)
                with col1:
                    custom_params['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 50)
                with col2:
                    custom_params['max_depth'] = st.slider("Max Depth", 3, 15, 6, 1)
                with col3:
                    custom_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            
            elif model_type == "LightGBM":
                col1, col2, col3 = st.columns(3)
                with col1:
                    custom_params['n_estimators'] = st.slider("N Estimators", 50, 500, 100, 50)
                with col2:
                    custom_params['num_leaves'] = st.slider("Num Leaves", 20, 100, 31, 5)
                with col3:
                    custom_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            
            elif model_type == "Logistic Regression":
                col1, col2 = st.columns(2)
                with col1:
                    custom_params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1)
                with col2:
                    custom_params['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000, 100)
            
            elif model_type == "SVM":
                col1, col2, col3 = st.columns(3)
                with col1:
                    custom_params['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, 0.1)
                with col2:
                    custom_params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                with col3:
                    custom_params['gamma'] = st.selectbox("Gamma", ['scale', 'auto'])
            
            elif model_type == "KNN":
                col1, col2 = st.columns(2)
                with col1:
                    custom_params['n_neighbors'] = st.slider("N Neighbors", 3, 30, 5, 2)
                with col2:
                    custom_params['weights'] = st.selectbox("Weights", ['uniform', 'distance'])
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("🎛️ Step 3: Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    with col3:
        experiment_name = st.text_input(
            "MLflow Experiment Name",
            f"Diabetes_{model_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"
        )
    
    st.markdown("---")
    
    # Train button
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        with st.spinner("🔄 Training model... This may take a few minutes."):
            try:
                # Preprocess data
                st.info("📊 Preprocessing data...")
                X_train, X_test, y_train, y_test, feature_names = preprocess_data(
                    data, test_size=test_size, random_state=random_state
                )
                
                st.success(f"✅ Data preprocessed: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
                
                # Train model
                st.info(f"🤖 Training {model_type}...")
                
                if model_type == "Ensemble (Best Performance)":
                    model, metrics, run_id = create_ensemble_model(
                        X_train, X_test, y_train, y_test, 
                        feature_names, experiment_name, random_state
                    )
                else:
                    model, metrics, run_id = train_model_with_mlflow(
                        model_type, X_train, X_test, y_train, y_test,
                        feature_names, experiment_name, custom_params, random_state
                    )
                
                st.success("✅ Model trained successfully!")
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Training Results")
                
                # Metrics cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['test_accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{metrics['test_precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['test_recall']:.4f}")
                with col4:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("F1-Score", f"{metrics['test_f1']:.4f}")
                with col2:
                    st.metric("Train Accuracy", f"{metrics['train_accuracy']:.4f}")
                with col3:
                    overfitting = metrics['train_accuracy'] - metrics['test_accuracy']
                    st.metric("Overfitting Gap", f"{overfitting:.4f}")
                
                # Visualizations
                st.markdown("#### 📈 Model Performance Visualizations")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fig = plot_confusion_matrix(
                        metrics['confusion_matrix'],
                        title=f"Confusion Matrix - {model_type}"
                    )
                    st.pyplot(fig)
                
                with col2:
                    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    if y_proba is not None:
                        fig = plot_roc_curve(y_test, y_proba, title=f"ROC Curve - {model_type}")
                        st.pyplot(fig)
                
                with col3:
                    if y_proba is not None:
                        fig = plot_precision_recall_curve(y_test, y_proba, title=f"PR Curve - {model_type}")
                        st.pyplot(fig)
                
                # MLflow info
                st.markdown("---")
                st.info(f"📁 **MLflow Run ID:** `{run_id}`\n\nExperiment: `{experiment_name}`")
                
                # Save model
                st.markdown("#### 💾 Save Model")
                
                model_filename = f"{model_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model_path = os.path.join("models", model_filename)
                
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)
                
                st.success(f"✅ Model saved to: `{model_path}`")
                
                # Download button
                with open(model_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Trained Model",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                # Store in session state for predictions
                st.session_state.active_model_path = model_path
                st.session_state.active_model_name = model_type
                
                st.success("🎉 Training complete! You can now make predictions with this model.")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)
