"""
Dataset Explorer Page
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def show():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>📊</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                Dataset Explorer
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Upload and explore your diabetes dataset with comprehensive EDA tools
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV File", 
        type=['csv'],
        help="Upload your diabetes dataset in CSV format"
    )
    
    # Load default dataset option
    col1, col2 = st.columns([3, 1])
    with col2:
        use_default = st.button("📂 Load Default Dataset", use_container_width=True)
    
    # Load data
    data = None
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return
    
    elif use_default:
        # Try to load default dataset
        try:
            import kagglehub
            path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
            csv_path = os.path.join(path, "diabetes.csv")
            data = pd.read_csv(csv_path)
            st.success(f"✅ Default dataset loaded! Shape: {data.shape}")
        except Exception as e:
            st.warning("⚠️ Could not auto-download. Trying local path...")
            try:
                data_path = os.path.join("data", "diabetes.csv")
                if os.path.exists(data_path):
                    data = pd.read_csv(data_path)
                    st.success(f"✅ Local dataset loaded! Shape: {data.shape}")
                else:
                    st.error("❌ No dataset found. Please upload a CSV file.")
                    return
            except Exception as e2:
                st.error(f"❌ Error: {str(e2)}")
                return
    
    if data is None:
        st.info("👆 Please upload a dataset or load the default one to begin exploration")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", "📊 Statistics", "📈 Distributions", "🔗 Correlations", "🔍 Missing Values"
    ])
    
    with tab1:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.2f} KB")
        with col4:
            if 'Outcome' in data.columns:
                st.metric("Target Classes", data['Outcome'].nunique())
        
        st.markdown("#### First 10 Rows")
        st.dataframe(data.head(10), use_container_width=True)
        
        st.markdown("#### Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes.values,
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab2:
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
        if 'Outcome' in data.columns:
            st.markdown("#### Target Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                data['Outcome'].value_counts().plot(kind='bar', ax=ax, color=['#3b82f6', '#ef4444'])
                ax.set_title('Target Distribution')
                ax.set_xlabel('Outcome')
                ax.set_ylabel('Count')
                plt.xticks(rotation=0)
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                data['Outcome'].value_counts().plot(
                    kind='pie', 
                    ax=ax, 
                    autopct='%1.1f%%',
                    colors=['#3b82f6', '#ef4444']
                )
                ax.set_ylabel('')
                ax.set_title('Target Distribution')
                st.pyplot(fig)
    
    with tab3:
        st.subheader("Feature Distributions")
        
        # Select columns to plot
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_cols = st.multiselect(
                "Select features to visualize",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if selected_cols:
                cols_per_row = 2
                n_rows = (len(selected_cols) + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(14, 4*n_rows))
                axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
                
                for idx, col in enumerate(selected_cols):
                    if idx < len(axes):
                        axes[idx].hist(data[col].dropna(), bins=30, color='#3b82f6', alpha=0.7, edgecolor='black')
                        axes[idx].set_title(f'{col} Distribution')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Frequency')
                
                # Hide extra subplots
                for idx in range(len(selected_cols), len(axes)):
                    axes[idx].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("No numeric columns found in the dataset")
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            # Correlation matrix
            corr_matrix = numeric_data.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Highly correlated features
            st.markdown("#### Highly Correlated Feature Pairs (|correlation| > 0.5)")
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.info("No highly correlated feature pairs found")
        else:
            st.warning("No numeric columns for correlation analysis")
    
    with tab5:
        st.subheader("Missing Values Analysis")
        
        missing = data.isnull().sum()
        missing_pct = (missing / len(data)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        }).sort_values('Missing Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if not missing_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#ef4444' if x > 40 else '#f59e0b' if x > 20 else '#10b981' 
                         for x in missing_df['Missing %']]
                bars = ax.barh(missing_df['Column'], missing_df['Missing Count'], color=colors)
                ax.set_xlabel('Missing Count')
                ax.set_title('Missing Values by Column')
                ax.grid(axis='x', alpha=0.3)
                
                # Add percentage labels
                for i, (bar, pct) in enumerate(zip(bars, missing_df['Missing %'])):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                           f'{pct:.1f}%', va='center', ha='left', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(missing_df.round(2), use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Check for zeros in critical columns
        st.markdown("#### Zero Values Check (Medical Impossibilities)")
        zero_check_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        zero_cols = [col for col in zero_check_cols if col in data.columns]
        
        if zero_cols:
            zero_counts = {}
            for col in zero_cols:
                zero_count = (data[col] == 0).sum()
                if zero_count > 0:
                    zero_counts[col] = {
                        'Count': zero_count,
                        'Percentage': (zero_count / len(data)) * 100
                    }
            
            if zero_counts:
                zero_df = pd.DataFrame(zero_counts).T
                zero_df.reset_index(inplace=True)
                zero_df.columns = ['Column', 'Zero Count', 'Zero %']
                st.warning("⚠️ Found zero values in medical columns (likely missing data):")
                st.dataframe(zero_df.round(2), use_container_width=True)
            else:
                st.success("✅ No suspicious zero values found")
    
    st.markdown("---")
    
    # Download processed data
    if st.button("💾 Download Dataset as CSV", use_container_width=True):
        csv = data.to_csv(index=False)
        st.download_button(
            label="⬇️ Click to Download",
            data=csv,
            file_name="diabetes_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )
