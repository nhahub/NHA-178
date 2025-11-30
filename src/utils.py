"""
Utility Functions for Pima Indians Diabetes ML Project
Helper functions, data versioning, logging, and visualization utilities
"""

import os
import json
import pickle
import hashlib
import time
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@contextmanager
def timer(title):
    """Context manager for timing code execution"""
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    print(f"{title} - done in {elapsed:.2f}s")


def save_model(model, filepath):
    """Save model to disk using pickle"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {filepath}")


def load_model(filepath):
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from: {filepath}")
    return model


def save_results(results, filepath):
    """Save results dictionary to JSON"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        else:
            return obj
    
    results_converted = convert_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=4)
    print(f"Results saved to: {filepath}")


def load_results(filepath):
    """Load results from JSON"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    print(f"Results loaded from: {filepath}")
    return results


def calculate_data_hash(data):
    """Calculate hash of dataset for versioning"""
    if isinstance(data, pd.DataFrame):
        data_bytes = pd.util.hash_pandas_object(data).values.tobytes()
    elif isinstance(data, np.ndarray):
        data_bytes = data.tobytes()
    else:
        data_bytes = str(data).encode()
    
    return hashlib.md5(data_bytes).hexdigest()


def create_data_version_info(data, description=""):
    """Create data version information"""
    version_info = {
        'timestamp': datetime.now().isoformat(),
        'hash': calculate_data_hash(data),
        'shape': data.shape if hasattr(data, 'shape') else None,
        'description': description
    }
    return version_info


def save_data_version(data, filepath, description=""):
    """Save data with version information"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save data
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, np.ndarray):
        np.save(filepath, data)
    
    # Save version info
    version_info = create_data_version_info(data, description)
    version_filepath = filepath + '.version.json'
    with open(version_filepath, 'w') as f:
        json.dump(version_info, f, indent=4)
    
    print(f"Data saved to: {filepath}")
    print(f"Version info saved to: {version_filepath}")


def setup_plotting_style():
    """Setup consistent plotting style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def create_summary_report(results_dict, output_path='model_summary_report.txt'):
    """Create a text summary report of all results"""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PIMA INDIANS DIABETES - MODEL EVALUATION SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Sort models by accuracy
        sorted_models = sorted(results_dict.items(), 
                             key=lambda x: x[1].get('accuracy', 0), 
                             reverse=True)
        
        f.write("MODEL RANKINGS (by Accuracy):\n")
        f.write("-"*80 + "\n")
        for rank, (model_name, metrics) in enumerate(sorted_models, 1):
            f.write(f"{rank}. {model_name}: {metrics.get('accuracy', 0):.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED METRICS FOR EACH MODEL:\n")
        f.write("="*80 + "\n\n")
        
        for model_name, metrics in sorted_models:
            f.write(f"\n{model_name.upper()}\n")
            f.write("-"*80 + "\n")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float, np.integer, np.floating)):
                    f.write(f"{metric_name:20s}: {metric_value:.4f}\n")
            f.write("\n")
        
        # Best model summary
        best_model_name, best_metrics = sorted_models[0]
        f.write("="*80 + "\n")
        f.write("BEST PERFORMING MODEL\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Accuracy: {best_metrics.get('accuracy', 0):.4f}\n")
        f.write(f"Precision: {best_metrics.get('precision', 0):.4f}\n")
        f.write(f"Recall: {best_metrics.get('recall', 0):.4f}\n")
        f.write(f"F1-Score: {best_metrics.get('f1_score', 0):.4f}\n")
        f.write(f"ROC AUC: {best_metrics.get('roc_auc', 0):.4f}\n")
        f.write("="*80 + "\n")
    
    print(f"Summary report saved to: {output_path}")
    return output_path


def print_section_header(title, width=60):
    """Print formatted section header"""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_results_table(results_df):
    """Print results DataFrame in a formatted way"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(results_df.to_string(index=False))
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')


def ensure_dir(directory):
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)
    return directory


def get_timestamp():
    """Get formatted timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def log_experiment_info(info_dict, filepath='experiment_info.json'):
    """Log experiment information to JSON file"""
    info_dict['timestamp'] = datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(info_dict, f, indent=4)
    
    print(f"Experiment info logged to: {filepath}")


class ExperimentLogger:
    """Logger for tracking experiment progress"""
    
    def __init__(self, log_file='experiment.log'):
        self.log_file = log_file
        self.start_time = time.time()
        
    def log(self, message, level='INFO'):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def log_metrics(self, model_name, metrics):
        """Log model metrics"""
        self.log(f"Model: {model_name}", level='METRICS')
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                self.log(f"  {metric_name}: {metric_value:.4f}", level='METRICS')
    
    def log_elapsed_time(self):
        """Log total elapsed time"""
        elapsed = time.time() - self.start_time
        self.log(f"Total elapsed time: {elapsed:.2f}s", level='TIME')


def plot_data_distribution(data, target_col, output_path='data_distribution.png'):
    """Plot target variable distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    data[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['lightskyblue', 'gold'])
    axes[0].set_title(f'Count of {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Pie chart
    data[target_col].value_counts().plot(kind='pie', ax=axes[1], colors=['lightskyblue', 'gold'], 
                                         autopct='%1.1f%%')
    axes[1].set_title(f'Distribution of {target_col}')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Data distribution plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Utility Functions Module")
    print("="*50)
    print("Available utilities:")
    print("- timer(): Context manager for timing")
    print("- save_model() / load_model(): Model persistence")
    print("- save_results() / load_results(): Results persistence")
    print("- create_data_version_info(): Data versioning")
    print("- create_summary_report(): Generate text reports")
    print("- ExperimentLogger: Experiment tracking")
    print("- Various helper functions for visualization and formatting")
