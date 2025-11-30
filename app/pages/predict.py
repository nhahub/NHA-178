"""
Prediction Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from app.utils.plots import plot_feature_importance_horizontal

def preprocess_for_prediction(input_data):
    """
    Preprocess input data for prediction (apply feature engineering)
    This creates the 16 engineered features to match the training data (24 total features)
    """
    df = input_data.copy()
    
    # Remove Outcome column if it exists (from uploaded CSV files)
    if 'Outcome' in df.columns:
        df = df.drop('Outcome', axis=1)
    
    # Handle missing values - replace 0 with NaN for certain features
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
    
    # Impute missing values using general medians (since we don't have Outcome for new data)
    # Using healthy medians as default (more conservative)
    if 'Insulin' in df.columns:
        df['Insulin'].fillna(102.5, inplace=True)
    if 'Glucose' in df.columns:
        df['Glucose'].fillna(107, inplace=True)
    if 'SkinThickness' in df.columns:
        df['SkinThickness'].fillna(27, inplace=True)
    if 'BloodPressure' in df.columns:
        df['BloodPressure'].fillna(70, inplace=True)
    if 'BMI' in df.columns:
        df['BMI'].fillna(30.1, inplace=True)
    
    # Create 16 engineered features (matching the notebook exactly)
    # Binary features
    df['N1'] = ((df['Age'] <= 30) & (df['Glucose'] <= 120)).astype(int)
    df['N2'] = (df['BMI'] <= 30).astype(int)
    df['N3'] = ((df['Age'] <= 30) & (df['Pregnancies'] <= 6)).astype(int)
    df['N4'] = ((df['Glucose'] <= 105) & (df['BloodPressure'] <= 80)).astype(int)
    df['N5'] = (df['SkinThickness'] <= 20).astype(int)
    df['N6'] = ((df['BMI'] < 30) & (df['SkinThickness'] <= 20)).astype(int)
    df['N7'] = ((df['Glucose'] <= 105) & (df['BMI'] <= 30)).astype(int)
    df['N9'] = (df['Insulin'] < 200).astype(int)
    df['N10'] = (df['BloodPressure'] < 80).astype(int)
    df['N11'] = ((df['Pregnancies'] < 4) & (df['Pregnancies'] != 0)).astype(int)
    
    # Continuous features
    df['N0'] = df['BMI'] * df['SkinThickness']
    df['N8'] = df['Pregnancies'] / df['Age']
    df['N13'] = df['Glucose'] / df['DiabetesPedigreeFunction']
    df['N12'] = df['Age'] * df['DiabetesPedigreeFunction']
    df['N14'] = df['Age'] / df['Insulin']
    df['N15'] = (df['N0'] < 1034).astype(int)
    
    return df

def show():
    # Page header with gradient
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0 1rem 0;'>
            <h1 style='font-size: 3rem; margin: 0;'>🔮 Make Predictions</h1>
            <p style='font-size: 1.2rem; color: #a78bfa; margin-top: 0.5rem;'>
                Get diabetes predictions using trained ML models
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model upload/selection section with glass card
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>📂 Step 1: Load Model</h2>
            <p style='color: #a78bfa;'>Select an active model or upload a pre-trained model file</p>
        </div>
    """, unsafe_allow_html=True)
    
    model = None
    model_name = "Unknown Model"
    
    # Option 1: Use active model from training
    if 'active_model_path' in st.session_state and st.session_state.active_model_path is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"📊 **Active Model:** {st.session_state.get('active_model_name', 'Trained Model')}")
        with col2:
            use_active = st.button("✅ Use This Model", use_container_width=True, type="primary")
        
        if use_active:
            try:
                model = joblib.load(st.session_state.active_model_path)
                model_name = st.session_state.get('active_model_name', 'Trained Model')
                st.success(f"✅ Model loaded: {model_name}")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                model = None
    
    # Option 2: Upload .pkl file
    st.markdown("**Or upload a model file:**")
    uploaded_model = st.file_uploader(
        "Upload .pkl model file (max 200MB)", 
        type=['pkl'], 
        key="model_upload",
        help="Upload a trained model saved as .pkl file"
    )
    
    if uploaded_model is not None and model is None:
        try:
            model = joblib.load(uploaded_model)
            model_name = uploaded_model.name
            st.success(f"✅ Model loaded: {model_name}")
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            model = None
    
    if model is None:
        st.markdown("""
            <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); 
                        border-radius: 12px; padding: 1.5rem; text-align: center; margin: 2rem 0;'>
                <h3 style='color: #fca5a5; margin: 0;'>⚠️ No Model Loaded</h3>
                <p style='color: #fecaca; margin-top: 0.5rem;'>
                    Please train a model first or upload a .pkl model file to make predictions
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("---")
    
    # Input method selection with glass card
    st.markdown("""
        <div class='glass-card'>
            <h2 style='margin-top: 0;'>📝 Step 2: Select Input Method</h2>
            <p style='color: #a78bfa;'>Choose how you want to provide patient data</p>
        </div>
    """, unsafe_allow_html=True)
    
    input_method = st.radio(
        "",
        ["🎛️ Manual Input (Sliders)", "📂 Upload CSV File"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if input_method == "🎛️ Manual Input (Sliders)":
        st.markdown("""
            <div class='glass-card'>
                <h2 style='margin-top: 0;'>🎛️ Enter Patient Information</h2>
                <p style='color: #a78bfa;'>Use the sliders below to input patient data</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                            border: 1px solid rgba(167, 139, 250, 0.3); margin-bottom: 1rem;'>
                    <h4 style='color: #a78bfa; margin: 0;'>👤 Basic Information</h4>
                </div>
            """, unsafe_allow_html=True)
            pregnancies = st.slider("👶 Pregnancies", 0, 17, 3, 1)
            age = st.slider("🎂 Age (years)", 21, 81, 33, 1)
            
            st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                            border: 1px solid rgba(167, 139, 250, 0.3); margin: 1rem 0;'>
                    <h4 style='color: #a78bfa; margin: 0;'>💉 Metabolic Indicators</h4>
                </div>
            """, unsafe_allow_html=True)
            glucose = st.slider("🩸 Glucose (mg/dL)", 0, 200, 120, 1)
            insulin = st.slider("💊 Insulin (μU/mL)", 0, 846, 79, 1)
            bmi = st.slider("⚖️ BMI", 0.0, 67.1, 32.0, 0.1)
        
        with col2:
            st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                            border: 1px solid rgba(167, 139, 250, 0.3); margin-bottom: 1rem;'>
                    <h4 style='color: #a78bfa; margin: 0;'>📏 Physical Measurements</h4>
                </div>
            """, unsafe_allow_html=True)
            blood_pressure = st.slider("❤️ Blood Pressure (mm Hg)", 0, 122, 72, 1)
            skin_thickness = st.slider("📐 Skin Thickness (mm)", 0, 99, 20, 1)
            
            st.markdown("""
                <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 12px; 
                            border: 1px solid rgba(167, 139, 250, 0.3); margin: 1rem 0;'>
                    <h4 style='color: #a78bfa; margin: 0;'>🧬 Genetic Factor</h4>
                </div>
            """, unsafe_allow_html=True)
            dpf = st.slider("🔬 Diabetes Pedigree Function", 0.078, 2.42, 0.47, 0.001)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predict button with animation
        if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
            make_prediction(model, input_data)
    
    else:  # CSV Upload
        st.markdown("""
            <div class='glass-card'>
                <h2 style='margin-top: 0;'>📂 Upload CSV File</h2>
                <p style='color: #a78bfa;'>Upload a CSV file containing patient data for batch predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_csv = st.file_uploader(
            "Upload CSV with patient data",
            type=['csv'],
            help="CSV should contain columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age",
            key="predict_csv"
        )
        
        if uploaded_csv is not None:
            try:
                input_data = pd.read_csv(uploaded_csv)
                st.success(f"✅ Loaded {len(input_data)} patient records")
                
                # Show preview
                st.markdown("#### Data Preview")
                st.dataframe(input_data.head(10), use_container_width=True)
                
                # Validate columns
                required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                
                missing_cols = [col for col in required_cols if col not in input_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    return
                
                # Predict button
                if st.button("🔮 Make Predictions", use_container_width=True, type="primary"):
                    make_batch_predictions(model, input_data)
                    
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")

def make_prediction(model, input_data):
    """Make single prediction and display results"""
    
    with st.spinner("🔄 Making prediction..."):
        try:
            # Preprocess input data (add engineered features)
            processed_data = preprocess_for_prediction(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0]
                prob_negative = proba[0] * 100
                prob_positive = proba[1] * 100
            else:
                prob_negative = None
                prob_positive = None
            
            st.markdown("---")
            
            # Result card with enhanced styling
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h2 style='color: #a78bfa; margin-bottom: 1rem;'>📊 Prediction Results</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(239, 68, 68, 0.9) 0%, rgba(220, 38, 38, 0.9) 100%); 
                                backdrop-filter: blur(10px);
                                padding: 3rem 2rem; 
                                border-radius: 20px; 
                                text-align: center; 
                                color: white;
                                box-shadow: 0 15px 40px rgba(239, 68, 68, 0.4);
                                border: 2px solid rgba(239, 68, 68, 0.5);
                                animation: pulse 2s ease-in-out infinite;'>
                        <h1 style='margin: 0; color: white; font-size: 3rem; text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>⚠️ DIABETIC</h1>
                        <p style='font-size: 1.3em; margin-top: 1rem; color: #fee2e2; font-weight: 500;'>
                            The model predicts diabetes presence
                        </p>
                    </div>
                    <style>
                        @keyframes pulse {
                            0%, 100% { transform: scale(1); }
                            50% { transform: scale(1.02); }
                        }
                    </style>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background: linear-gradient(135deg, rgba(16, 185, 129, 0.9) 0%, rgba(5, 150, 105, 0.9) 100%); 
                                backdrop-filter: blur(10px);
                                padding: 3rem 2rem; 
                                border-radius: 20px; 
                                text-align: center; 
                                color: white;
                                box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4);
                                border: 2px solid rgba(16, 185, 129, 0.5);
                                animation: pulse 2s ease-in-out infinite;'>
                        <h1 style='margin: 0; color: white; font-size: 3rem; text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>✅ HEALTHY</h1>
                        <p style='font-size: 1.3em; margin-top: 1rem; color: #d1fae5; font-weight: 500;'>
                            The model predicts no diabetes
                        </p>
                    </div>
                    <style>
                        @keyframes pulse {
                            0%, 100% { transform: scale(1); }
                            50% { transform: scale(1.02); }
                        }
                    </style>
                """, unsafe_allow_html=True)
            
            # Probability display
            if prob_positive is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='text-align: center;'>
                        <h3 style='color: #a78bfa;'>📈 Prediction Confidence</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Healthy Probability",
                        f"{prob_negative:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Diabetic Probability", 
                        f"{prob_positive:.2f}%",
                        delta=None
                    )
                
                # Enhanced progress bar
                st.markdown(f"""
                    <div style='margin: 1.5rem 0;'>
                        <div style='background: rgba(26, 26, 46, 0.6); border-radius: 20px; height: 30px; overflow: hidden; border: 1px solid rgba(167, 139, 250, 0.3);'>
                            <div style='background: linear-gradient(90deg, #10b981 0%, #ef4444 100%); 
                                        width: {prob_positive}%; 
                                        height: 100%; 
                                        transition: width 1s ease;
                                        display: flex;
                                        align-items: center;
                                        justify-content: center;
                                        color: white;
                                        font-weight: 600;'>
                                {prob_positive:.1f}%
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='text-align: center;'>
                        <h3 style='color: #a78bfa;'>📊 Feature Importance</h3>
                    </div>
                """, unsafe_allow_html=True)
                feature_names = input_data.columns.tolist()
                importances = model.feature_importances_
                
                # Set dark theme for matplotlib
                plt.style.use('dark_background')
                fig = plot_feature_importance_horizontal(feature_names, importances)
                fig.patch.set_facecolor('none')
                st.pyplot(fig)
            
            # Input summary with styled expander
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 View Input Summary", expanded=False):
                st.dataframe(input_data.T, use_container_width=True)
            
            st.success("✅ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

def make_batch_predictions(model, input_data):
    """Make batch predictions and display results"""
    
    with st.spinner("🔄 Making predictions..."):
        try:
            # Preprocess input data (add engineered features)
            processed_data = preprocess_for_prediction(input_data)
            
            # Make predictions
            predictions = model.predict(processed_data)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(processed_data)
                prob_positive = probabilities[:, 1] * 100
            else:
                prob_positive = None
            
            # Create results dataframe
            results = input_data.copy()
            results['Prediction'] = ['Diabetic' if p == 1 else 'Healthy' for p in predictions]
            results['Prediction_Numeric'] = predictions
            
            if prob_positive is not None:
                results['Diabetes_Probability_%'] = prob_positive.round(2)
            
            st.markdown("---")
            
            # Header with animation
            st.markdown("""
                <div style='text-align: center; margin: 2rem 0;'>
                    <h2 style='color: #a78bfa; font-size: 2.5rem;'>📊 Batch Prediction Results</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Summary statistics with enhanced cards
            col1, col2, col3, col4 = st.columns(4)
            
            diabetic_count = (predictions == 1).sum()
            healthy_count = (predictions == 0).sum()
            diabetic_pct = (diabetic_count / len(results)) * 100
            
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>👥 {len(results)}</div>
                        <div class='metric-label'>Total Patients</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class='metric-card' style='border-color: rgba(239, 68, 68, 0.4);'>
                        <div class='metric-value' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>⚠️ {diabetic_count}</div>
                        <div class='metric-label'>Diabetic</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                    <div class='metric-card' style='border-color: rgba(16, 185, 129, 0.4);'>
                        <div class='metric-value' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>✅ {healthy_count}</div>
                        <div class='metric-label'>Healthy</div>
                    </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-value'>📈 {diabetic_pct:.1f}%</div>
                        <div class='metric-label'>Diabetic Rate</div>
                    </div>
                """, unsafe_allow_html=True)
            
            
            # Results table with header
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
                <div style='text-align: center;'>
                    <h3 style='color: #a78bfa;'>📋 Detailed Results</h3>
                </div>
            """, unsafe_allow_html=True)
            st.dataframe(results, use_container_width=True)
            
            # Visualization with dark theme
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                prediction_counts = results['Prediction'].value_counts()
                colors = ['#10b981' if x == 'Healthy' else '#ef4444' for x in prediction_counts.index]
                bars = ax.bar(prediction_counts.index, prediction_counts.values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
                
                ax.set_title('Prediction Distribution', color='#a78bfa', fontsize=14, fontweight='bold', pad=15)
                ax.set_ylabel('Count', color='#d1d5db', fontsize=11)
                ax.tick_params(colors='#d1d5db')
                ax.grid(axis='y', alpha=0.2, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#a78bfa')
                ax.spines['bottom'].set_color('#a78bfa')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', color='white', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                if prob_positive is not None:
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(6, 4))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    n, bins, patches = ax.hist(prob_positive, bins=20, color='#667eea', alpha=0.8, edgecolor='white', linewidth=1.5)
                    
                    # Color gradient for histogram
                    for i, patch in enumerate(patches):
                        patch.set_facecolor(plt.cm.RdYlGn_r(bins[i]/100))
                    
                    ax.set_title('Diabetes Probability Distribution', color='#a78bfa', fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('Probability (%)', color='#d1d5db', fontsize=11)
                    ax.set_ylabel('Frequency', color='#d1d5db', fontsize=11)
                    ax.tick_params(colors='#d1d5db')
                    ax.grid(axis='y', alpha=0.2, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#a78bfa')
                    ax.spines['bottom'].set_color('#a78bfa')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Download results with enhanced button
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            csv = results.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            
            st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); 
                            border-radius: 12px; padding: 1rem; text-align: center; margin-top: 1rem;'>
                    <p style='color: #6ee7b7; margin: 0; font-weight: 600;'>✅ Batch predictions completed successfully!</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Batch prediction failed: {str(e)}")
            st.exception(e)
