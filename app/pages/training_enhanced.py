"""
Enhanced Model Training Page with Two-Stage Optimization
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
from sklearn.model_selection import GridSearchCV

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.utils.model_utils import (
    get_model, preprocess_data, train_model_with_mlflow,
    create_ensemble_model
)
from app.utils.plots import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve

def collect_comprehensive_metrics(model, X_train, y_train, X_test, y_test, model_name):
    """Collect all training and testing metrics for comprehensive analysis"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Training predictions
    train_pred = model.predict(X_train)
    train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Testing predictions  
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Comprehensive metrics
    metrics = {
        'model_name': model_name,
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred, zero_division=0),
        'test_precision': precision_score(y_test, test_pred, zero_division=0),
        'train_recall': recall_score(y_train, train_pred, zero_division=0),
        'test_recall': recall_score(y_test, test_pred, zero_division=0),
        'train_f1': f1_score(y_train, train_pred, zero_division=0),
        'test_f1': f1_score(y_test, test_pred, zero_division=0),
    }
    
    if train_proba is not None:
        metrics['train_roc_auc'] = roc_auc_score(y_train, train_proba)
    if test_proba is not None:
        metrics['test_roc_auc'] = roc_auc_score(y_test, test_proba)
        
    return metrics

def get_param_grid(model_type):
    """Get optimized parameter grid for GridSearchCV"""
    
    grids = {
        "LightGBM": {
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [200, 400, 600],
            'num_leaves': [31, 63],
            'max_depth': [7, 9],
            'subsample': [0.8, 1.0],
            'reg_lambda': [1.0, 5.0],
        },
        "XGBoost": {
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [200, 400, 600],
            'max_depth': [5, 7, 9],
            'subsample': [0.8, 1.0],
            'gamma': [0, 0.1],
            'reg_lambda': [1.0, 5.0],
        },
        "Random Forest": {
            'n_estimators': [200, 400],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'max_features': ['sqrt', None],
        },
        "Gradient Boosting": {
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [200, 400],
            'max_depth': [5, 7],
            'subsample': [0.8, 1.0],
        }
    }
    
    return grids.get(model_type, {})

def show():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>🔧</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                Advanced Model Training
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Two-Stage Optimization Pipeline for Maximum Performance
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Training strategy selector
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>🎯 Select Training Strategy</h2>
            <p style='color: #a78bfa;'>Choose between quick baseline or comprehensive optimization</p>
        </div>
    """, unsafe_allow_html=True)
    
    training_mode = st.radio(
        "",
        ["⚡ Stage 1: Quick Baseline (All 8 Models)", "🎯 Stage 2: Optimize Top Performers (GridSearchCV)", "🚀 Full Pipeline (Both Stages)"],
        help="Stage 1 trains all models quickly. Stage 2 optimizes top 2 performers with GridSearchCV."
    )
    
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
        
        if 'Outcome' not in data.columns:
            st.error("❌ Dataset must contain 'Outcome' column")
            return
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("🎛️ Step 2: Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    with col3:
        cv_folds = st.slider("CV Folds (Stage 2)", 3, 10, 5, 1)
    
    st.markdown("---")
    
    # Train button
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        
        # Preprocess data
        with st.spinner("📊 Preprocessing data..."):
            try:
                X_train, X_test, y_train, y_test, feature_names = preprocess_data(
                    data, test_size=test_size, random_state=random_state
                )
                st.success(f"✅ Data preprocessed: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            except Exception as e:
                st.error(f"❌ Preprocessing failed: {str(e)}")
                return
        
        st.markdown("---")
        
        # Stage 1: Quick Baseline
        if training_mode in ["⚡ Stage 1: Quick Baseline (All 8 Models)", "🚀 Full Pipeline (Both Stages)"]:
            st.markdown("""
                <div style='text-align: center; background: rgba(102, 126, 234, 0.1); padding: 1.5rem; border-radius: 12px; margin: 2rem 0;'>
                    <h2 style='color: #a78bfa; margin: 0;'>⚡ STAGE 1: QUICK BASELINE</h2>
                    <p style='color: #c4b5fd; margin-top: 0.5rem;'>Training all 8 algorithms with default parameters</p>
                </div>
            """, unsafe_allow_html=True)
            
            # List of all 8 models
            model_list = [
                "Logistic Regression", "Random Forest", "XGBoost", "LightGBM",
                "SVM", "Gradient Boosting", "KNN", "Decision Tree"
            ]
            
            baseline_results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, model_name in enumerate(model_list):
                status_text.text(f"Training {model_name}... ({idx+1}/{len(model_list)})")
                
                try:
                    model = get_model(model_name, random_state)
                    model.fit(X_train, y_train)
                    metrics = collect_comprehensive_metrics(model, X_train, y_train, X_test, y_test, model_name)
                    baseline_results[model_name] = metrics
                    
                except Exception as e:
                    st.warning(f"⚠️ {model_name} failed: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(model_list))
            
            status_text.empty()
            progress_bar.empty()
            
            st.success(f"✅ Stage 1 Complete! Trained {len(baseline_results)} models")
            
            # Display Stage 1 results
            st.markdown("#### 📊 Stage 1 Results")
            
            results_df = pd.DataFrame([
                {
                    'Model': name,
                    'Train Acc': f"{metrics['train_accuracy']:.4f}",
                    'Test Acc': f"{metrics['test_accuracy']:.4f}",
                    'Test F1': f"{metrics['test_f1']:.4f}",
                    'Test ROC-AUC': f"{metrics.get('test_roc_auc', 0):.4f}",
                    'Overfitting': f"{metrics['train_accuracy'] - metrics['test_accuracy']:.4f}"
                }
                for name, metrics in baseline_results.items()
            ]).sort_values('Test Acc', ascending=False)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Get top 2 performers
            top_2 = results_df.head(2)['Model'].tolist()
            
            st.info(f"🏆 **Top 2 Performers:** {', '.join(top_2)}")
            
            # Store in session state for Stage 2
            st.session_state.baseline_results = baseline_results
            st.session_state.top_2_models = top_2
            st.session_state.train_data = (X_train, X_test, y_train, y_test, feature_names)
            
            st.markdown("---")
        
        # Stage 2: Optimize Top Performers
        if training_mode in ["🎯 Stage 2: Optimize Top Performers (GridSearchCV)", "🚀 Full Pipeline (Both Stages)"]:
            
            # Check if Stage 1 was run
            if 'top_2_models' not in st.session_state:
                st.warning("⚠️ Please run Stage 1 first to identify top performers")
                return
            
            st.markdown("""
                <div style='text-align: center; background: rgba(118, 75, 162, 0.1); padding: 1.5rem; border-radius: 12px; margin: 2rem 0;'>
                    <h2 style='color: #a78bfa; margin: 0;'>🎯 STAGE 2: HYPERPARAMETER OPTIMIZATION</h2>
                    <p style='color: #c4b5fd; margin-top: 0.5rem;'>GridSearchCV on top 2 performers</p>
                </div>
            """, unsafe_allow_html=True)
            
            top_2_models = st.session_state.top_2_models
            X_train, X_test, y_train, y_test, feature_names = st.session_state.train_data
            
            st.info(f"🎯 Optimizing: **{', '.join(top_2_models)}**")
            
            optimized_results = {}
            
            for model_name in top_2_models:
                st.markdown(f"#### Optimizing {model_name}")
                
                param_grid = get_param_grid(model_name)
                
                if not param_grid:
                    st.warning(f"⚠️ No parameter grid defined for {model_name}")
                    continue
                
                # Calculate search space
                search_space = np.prod([len(v) for v in param_grid.values()])
                st.info(f"🔍 Search space: **{search_space}** combinations")
                
                with st.spinner(f"Running GridSearchCV on {model_name}..."):
                    try:
                        base_model = get_model(model_name, random_state)
                        
                        grid_search = GridSearchCV(
                            base_model,
                            param_grid,
                            cv=cv_folds,
                            scoring='accuracy',
                            n_jobs=-1,
                            verbose=0
                        )
                        
                        grid_search.fit(X_train, y_train)
                        
                        # Get best model and metrics
                        best_model = grid_search.best_estimator_
                        metrics = collect_comprehensive_metrics(best_model, X_train, y_train, X_test, y_test, model_name)
                        metrics['best_params'] = grid_search.best_params_
                        metrics['cv_score'] = grid_search.best_score_
                        
                        optimized_results[model_name] = metrics
                        
                        st.success(f"✅ {model_name} optimized!")
                        st.json(grid_search.best_params_)
                        
                    except Exception as e:
                        st.error(f"❌ Optimization failed for {model_name}: {str(e)}")
            
            # Display optimization results
            if optimized_results:
                st.markdown("---")
                st.markdown("#### 📊 Stage 2 Results (Optimized)")
                
                # Comparison table
                comparison_data = []
                
                for model_name in top_2_models:
                    if model_name in st.session_state.baseline_results:
                        baseline = st.session_state.baseline_results[model_name]
                        optimized = optimized_results.get(model_name)
                        
                        if optimized:
                            comparison_data.append({
                                'Model': model_name,
                                'Baseline Test Acc': f"{baseline['test_accuracy']:.4f}",
                                'Optimized Test Acc': f"{optimized['test_accuracy']:.4f}",
                                'Improvement': f"{(optimized['test_accuracy'] - baseline['test_accuracy']) * 100:.2f}%",
                                'Optimized F1': f"{optimized['test_f1']:.4f}",
                                'Optimized ROC-AUC': f"{optimized.get('test_roc_auc', 0):.4f}"
                            })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Identify best model
                    best_model_name = max(optimized_results.keys(), 
                                        key=lambda k: optimized_results[k]['test_accuracy'])
                    
                    st.success(f"🏆 **Best Optimized Model:** {best_model_name} - Test Accuracy: {optimized_results[best_model_name]['test_accuracy']:.4f}")
        
        st.markdown("---")
        st.success("🎉 Training pipeline completed!")

