"""
MLflow Model Explorer Page
"""

import streamlit as st
import mlflow
import pandas as pd
import os
from datetime import datetime

def show():
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: rgba(102, 126, 234, 0.05); 
                    border-radius: 15px; margin-bottom: 2rem;'>
            <div style='font-size: 4rem; margin-bottom: 1rem; filter: drop-shadow(0 0 15px rgba(102, 126, 234, 0.8));'>📁</div>
            <h1 style='font-size: 3em; margin: 0; color: #ffffff; font-weight: 700; letter-spacing: 0.5px; text-shadow: 0 2px 15px rgba(102, 126, 234, 0.8);'>
                MLflow Model Explorer
            </h1>
            <p style='font-size: 1.2em; color: #c4b5fd; margin-top: 1rem; font-weight: 500;'>
                Browse and manage your MLflow experiments and models
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # MLflow tracking URI
    tracking_uri = st.text_input(
        "MLflow Tracking URI",
        value="file:./mlruns",
        help="Path to MLflow tracking directory"
    )
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        st.success(f"✅ Connected to MLflow at: `{tracking_uri}`")
    except Exception as e:
        st.error(f"❌ Failed to connect to MLflow: {str(e)}")
        return
    
    st.markdown("---")
    
    # Get all experiments
    try:
        experiments = client.search_experiments()
        
        if not experiments:
            st.warning("⚠️ No experiments found. Train a model first!")
            return
        
        # Experiment selector
        exp_names = [exp.name for exp in experiments if exp.name != "Default"]
        
        if not exp_names:
            st.warning("⚠️ No experiments found (excluding Default)")
            return
        
        selected_exp_name = st.selectbox(
            "Select Experiment",
            exp_names,
            help="Choose an experiment to view its runs"
        )
        
        # Get selected experiment
        selected_exp = next(exp for exp in experiments if exp.name == selected_exp_name)
        
        # Display experiment info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Experiment ID", selected_exp.experiment_id)
        with col2:
            # Count runs
            runs = client.search_runs(selected_exp.experiment_id)
            st.metric("Total Runs", len(runs))
        with col3:
            st.metric("Lifecycle Stage", selected_exp.lifecycle_stage)
        
        st.markdown("---")
        
        # Get runs for selected experiment
        runs = client.search_runs(
            experiment_ids=[selected_exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=100
        )
        
        if not runs:
            st.warning(f"⚠️ No runs found in experiment '{selected_exp_name}'")
            return
        
        st.subheader(f"📊 Runs in '{selected_exp_name}'")
        st.markdown(f"Found **{len(runs)}** runs")
        
        # Create runs dataframe
        runs_data = []
        for run in runs:
            run_info = {
                'Run ID': run.info.run_id[:8] + "...",
                'Full Run ID': run.info.run_id,
                'Status': run.info.status,
                'Start Time': datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'Duration (s)': f"{(run.info.end_time - run.info.start_time) / 1000:.2f}" if run.info.end_time else "Running",
            }
            
            # Add metrics
            if run.data.metrics:
                run_info['Accuracy'] = run.data.metrics.get('test_accuracy', 'N/A')
                run_info['ROC-AUC'] = run.data.metrics.get('roc_auc', 'N/A')
                run_info['F1-Score'] = run.data.metrics.get('test_f1', 'N/A')
            
            # Add params
            if run.data.params:
                run_info['Model'] = run.data.params.get('model_name', run.data.params.get('model_type', 'N/A'))
            
            runs_data.append(run_info)
        
        runs_df = pd.DataFrame(runs_data)
        
        # Display runs table
        st.dataframe(
            runs_df.drop('Full Run ID', axis=1) if 'Full Run ID' in runs_df.columns else runs_df,
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Select a run to view details
        st.subheader("🔍 Run Details")
        
        selected_run_idx = st.selectbox(
            "Select a run to view details",
            range(len(runs)),
            format_func=lambda i: f"{runs_data[i]['Run ID']} - {runs_data[i].get('Model', 'Unknown')} - Acc: {runs_data[i].get('Accuracy', 'N/A')}"
        )
        
        selected_run = runs[selected_run_idx]
        
        # Display run details in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "⚙️ Parameters", "📁 Artifacts", "🎯 Actions"])
        
        with tab1:
            st.markdown("#### Metrics")
            if selected_run.data.metrics:
                metrics_df = pd.DataFrame([selected_run.data.metrics]).T
                metrics_df.columns = ['Value']
                metrics_df['Metric'] = metrics_df.index
                metrics_df = metrics_df[['Metric', 'Value']].sort_values('Metric')
                
                # Display as cards
                cols = st.columns(4)
                important_metrics = ['test_accuracy', 'roc_auc', 'test_precision', 'test_recall']
                
                for idx, metric in enumerate(important_metrics):
                    if metric in selected_run.data.metrics:
                        with cols[idx]:
                            st.metric(
                                metric.replace('test_', '').replace('_', ' ').title(),
                                f"{selected_run.data.metrics[metric]:.4f}"
                            )
                
                st.markdown("##### All Metrics")
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No metrics logged for this run")
        
        with tab2:
            st.markdown("#### Parameters")
            if selected_run.data.params:
                params_df = pd.DataFrame([selected_run.data.params]).T
                params_df.columns = ['Value']
                params_df['Parameter'] = params_df.index
                params_df = params_df[['Parameter', 'Value']].sort_values('Parameter')
                st.dataframe(params_df, use_container_width=True)
            else:
                st.info("No parameters logged for this run")
        
        with tab3:
            st.markdown("#### Artifacts")
            try:
                artifacts = client.list_artifacts(selected_run.info.run_id)
                
                if artifacts:
                    artifact_list = []
                    for artifact in artifacts:
                        artifact_list.append({
                            'Path': artifact.path,
                            'Size (KB)': f"{artifact.file_size / 1024:.2f}" if artifact.file_size else "N/A",
                            'Type': 'Directory' if artifact.is_dir else 'File'
                        })
                    
                    artifacts_df = pd.DataFrame(artifact_list)
                    st.dataframe(artifacts_df, use_container_width=True)
                    
                    # Download artifact
                    selected_artifact = st.selectbox(
                        "Select artifact to download",
                        [a.path for a in artifacts if not a.is_dir]
                    )
                    
                    if st.button("⬇️ Download Artifact", use_container_width=True):
                        try:
                            artifact_path = client.download_artifacts(
                                selected_run.info.run_id,
                                selected_artifact
                            )
                            
                            with open(artifact_path, 'rb') as f:
                                st.download_button(
                                    label=f"💾 Download {os.path.basename(selected_artifact)}",
                                    data=f,
                                    file_name=os.path.basename(selected_artifact),
                                    mime="application/octet-stream",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.error(f"❌ Error downloading artifact: {str(e)}")
                else:
                    st.info("No artifacts logged for this run")
            except Exception as e:
                st.error(f"❌ Error listing artifacts: {str(e)}")
        
        with tab4:
            st.markdown("#### Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Set as Active Model", use_container_width=True, type="primary"):
                    try:
                        # Try to load model from artifacts
                        model_path = None
                        artifacts = client.list_artifacts(selected_run.info.run_id)
                        
                        for artifact in artifacts:
                            if 'model' in artifact.path.lower() and artifact.path.endswith('.pkl'):
                                model_path = client.download_artifacts(
                                    selected_run.info.run_id,
                                    artifact.path
                                )
                                break
                        
                        if model_path:
                            st.session_state.active_model_path = model_path
                            st.session_state.active_model_name = selected_run.data.params.get('model_name', 'MLflow Model')
                            st.success(f"✅ Model activated! You can now use it for predictions.")
                        else:
                            st.warning("⚠️ No model artifact found in this run")
                    except Exception as e:
                        st.error(f"❌ Error activating model: {str(e)}")
            
            with col2:
                if st.button("🗑️ Delete Run", use_container_width=True):
                    try:
                        client.delete_run(selected_run.info.run_id)
                        st.success("✅ Run deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting run: {str(e)}")
            
            st.markdown("---")
            
            st.markdown("#### Run Information")
            run_info_data = {
                'Run ID': selected_run.info.run_id,
                'Experiment ID': selected_run.info.experiment_id,
                'Status': selected_run.info.status,
                'Start Time': datetime.fromtimestamp(selected_run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                'End Time': datetime.fromtimestamp(selected_run.info.end_time / 1000).strftime('%Y-%m-%d %H:%M:%S') if selected_run.info.end_time else 'N/A',
                'Artifact URI': selected_run.info.artifact_uri,
            }
            
            for key, value in run_info_data.items():
                st.text(f"{key}: {value}")
        
        st.markdown("---")
        
        # Comparison section
        st.subheader("📊 Compare Runs")
        
        if len(runs) > 1:
            compare_indices = st.multiselect(
                "Select runs to compare",
                range(len(runs)),
                default=[0, 1] if len(runs) > 1 else [0],
                format_func=lambda i: f"{runs_data[i]['Run ID']} - {runs_data[i].get('Model', 'Unknown')}"
            )
            
            if len(compare_indices) > 1:
                compare_data = []
                for idx in compare_indices:
                    run = runs[idx]
                    data = {
                        'Run': runs_data[idx]['Run ID'],
                        'Model': run.data.params.get('model_name', 'N/A'),
                        'Accuracy': run.data.metrics.get('test_accuracy', 'N/A'),
                        'ROC-AUC': run.data.metrics.get('roc_auc', 'N/A'),
                        'F1-Score': run.data.metrics.get('test_f1', 'N/A'),
                        'Precision': run.data.metrics.get('test_precision', 'N/A'),
                        'Recall': run.data.metrics.get('test_recall', 'N/A'),
                    }
                    compare_data.append(data)
                
                compare_df = pd.DataFrame(compare_data)
                st.dataframe(compare_df, use_container_width=True)
                
                # Visualization
                if len(compare_indices) <= 5:
                    import matplotlib.pyplot as plt
                    
                    metrics_to_plot = ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
                    metrics_to_plot = [m for m in metrics_to_plot if m in compare_df.columns]
                    
                    if metrics_to_plot:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        compare_df.set_index('Run')[metrics_to_plot].plot(kind='bar', ax=ax)
                        ax.set_title('Model Comparison')
                        ax.set_ylabel('Score')
                        ax.legend(loc='best')
                        ax.grid(axis='y', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.info("Train more models to enable comparison")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)
