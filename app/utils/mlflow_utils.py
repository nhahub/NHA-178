"""
MLflow Utilities for Experiment Tracking
"""

import mlflow
import mlflow.sklearn
from pathlib import Path

def get_mlflow_tracking_uri():
    """
    Get default MLflow tracking URI
    
    Returns:
        MLflow tracking URI path
    """
    return str(Path(__file__).parent.parent.parent / "mlruns")

def set_mlflow_tracking_uri(uri=None):
    """
    Set MLflow tracking URI
    
    Args:
        uri: Custom tracking URI (defaults to local mlruns folder)
    """
    if uri is None:
        uri = get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(uri)

def get_all_experiments():
    """
    Get all MLflow experiments
    
    Returns:
        List of experiment dictionaries
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    return [
        {
            'experiment_id': exp.experiment_id,
            'name': exp.name,
            'artifact_location': exp.artifact_location,
            'lifecycle_stage': exp.lifecycle_stage
        }
        for exp in experiments
    ]

def get_experiment_runs(experiment_id):
    """
    Get all runs for an experiment
    
    Args:
        experiment_id: MLflow experiment ID
    
    Returns:
        List of run dictionaries
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    run_data = []
    for run in runs:
        run_dict = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
            'metrics': run.data.metrics,
            'params': run.data.params,
            'tags': run.data.tags
        }
        run_data.append(run_dict)
    
    return run_data

def get_run_details(run_id):
    """
    Get detailed information for a specific run
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        Dictionary with run details
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    return {
        'run_id': run.info.run_id,
        'experiment_id': run.info.experiment_id,
        'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
        'status': run.info.status,
        'start_time': run.info.start_time,
        'end_time': run.info.end_time,
        'artifact_uri': run.info.artifact_uri,
        'metrics': run.data.metrics,
        'params': run.data.params,
        'tags': run.data.tags
    }

def get_run_artifacts(run_id):
    """
    List artifacts for a run
    
    Args:
        run_id: MLflow run ID
    
    Returns:
        List of artifact paths
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    
    return [
        {
            'path': artifact.path,
            'is_dir': artifact.is_dir,
            'file_size': artifact.file_size
        }
        for artifact in artifacts
    ]

def load_model_from_run(run_id, model_path="model"):
    """
    Load a model from MLflow run
    
    Args:
        run_id: MLflow run ID
        model_path: Path to model artifact (default: "model")
    
    Returns:
        Loaded model
    """
    set_mlflow_tracking_uri()
    model_uri = f"runs:/{run_id}/{model_path}"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def compare_runs(run_ids, metrics=['test_accuracy', 'test_f1', 'roc_auc']):
    """
    Compare metrics across multiple runs
    
    Args:
        run_ids: List of MLflow run IDs
        metrics: List of metric names to compare
    
    Returns:
        DataFrame with comparison data
    """
    import pandas as pd
    
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    
    comparison_data = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        row = {
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'model_name': run.data.params.get('model_name', 'Unknown')
        }
        
        for metric in metrics:
            row[metric] = run.data.metrics.get(metric, None)
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def delete_run(run_id):
    """
    Delete a specific run
    
    Args:
        run_id: MLflow run ID
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    client.delete_run(run_id)

def set_run_tag(run_id, key, value):
    """
    Set a tag for a run
    
    Args:
        run_id: MLflow run ID
        key: Tag key
        value: Tag value
    """
    set_mlflow_tracking_uri()
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, key, value)
