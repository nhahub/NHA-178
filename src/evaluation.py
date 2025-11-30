"""
Evaluation Module for Pima Indians Diabetes Classification
Comprehensive metrics calculation, visualization, and artifact logging
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import mlflow


class ModelEvaluator:
    """Comprehensive model evaluation and metrics calculation"""
    
    def __init__(self, artifact_dir='artifacts'):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        
    def calculate_metrics(self, y_true, y_pred, y_proba=None):
        """Calculate all classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Create and save confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, f'{model_name}_confusion_matrix.png')
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_roc_curve(self, y_true, y_proba, model_name, save_path=None):
        """Create and save ROC curve plot"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, f'{model_name}_roc_curve.png')
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_precision_recall_curve(self, y_true, y_proba, model_name, save_path=None):
        """Create and save precision-recall curve plot"""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='red', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, f'{model_name}_pr_curve.png')
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20, save_path=None):
        """Plot and save feature importance (if available)"""
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, f'{model_name}_feature_importance.png')
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_metrics_comparison(self, metrics_dict, save_path=None):
        """Create bar plot comparing key metrics"""
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        metrics_values = [
            metrics_dict.get('accuracy', 0),
            metrics_dict.get('precision', 0),
            metrics_dict.get('recall', 0),
            metrics_dict.get('f1_score', 0),
            metrics_dict.get('roc_auc', 0)
        ]
        
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=colors)
        plt.ylabel('Score')
        plt.ylim([0, 1])
        plt.title('Performance Metrics Overview')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, 'metrics_comparison.png')
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_classification_report(self, y_true, y_pred, model_name, save_path=None):
        """Generate and save classification report"""
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, f'{model_name}_classification_report.csv')
        
        report_df.to_csv(save_path)
        
        return save_path, report
    
    def evaluate_model(self, model, X_test, y_test, model_name, feature_names=None, log_to_mlflow=True):
        """Complete model evaluation with all metrics and visualizations"""
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate visualizations
        artifacts = {}
        
        # Confusion matrix
        cm_path = self.plot_confusion_matrix(y_test, y_pred, model_name)
        artifacts['confusion_matrix'] = cm_path
        
        # ROC curve (if probabilities available)
        if y_proba is not None:
            roc_path = self.plot_roc_curve(y_test, y_proba, model_name)
            artifacts['roc_curve'] = roc_path
            
            # Precision-Recall curve
            pr_path = self.plot_precision_recall_curve(y_test, y_proba, model_name)
            artifacts['pr_curve'] = pr_path
        
        # Feature importance (if available)
        if feature_names is not None:
            fi_path = self.plot_feature_importance(model, feature_names, model_name)
            if fi_path:
                artifacts['feature_importance'] = fi_path
        
        # Metrics comparison
        metrics_comp_path = self.plot_metrics_comparison(metrics)
        artifacts['metrics_comparison'] = metrics_comp_path
        
        # Classification report
        report_path, report_dict = self.generate_classification_report(y_test, y_pred, model_name)
        artifacts['classification_report'] = report_path
        
        # Log to MLflow if enabled
        if log_to_mlflow:
            self.log_to_mlflow(metrics, artifacts)
        
        return metrics, artifacts, report_dict
    
    def log_to_mlflow(self, metrics, artifacts):
        """Log metrics and artifacts to MLflow"""
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                mlflow.log_metric(metric_name, float(metric_value))
        
        # Log artifacts
        for artifact_name, artifact_path in artifacts.items():
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
    
    def compare_models(self, results_dict, save_path=None):
        """Compare multiple models and create comparison visualizations"""
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'ROC AUC': metrics.get('roc_auc', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        
        # Save comparison table
        if save_path is None:
            save_path = os.path.join(self.artifact_dir, 'model_comparison.csv')
        comparison_df.to_csv(save_path, index=False)
        
        # Create comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy comparison
        axes[0].barh(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
        axes[0].set_xlabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_xlim([0, 1])
        for i, v in enumerate(comparison_df['Accuracy']):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # All metrics comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        x = np.arange(len(comparison_df))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - 2) * width
            axes[1].bar(x + offset, comparison_df[metric], width, label=metric)
        
        axes[1].set_ylabel('Score')
        axes[1].set_title('Comprehensive Metrics Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        axes[1].legend()
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        viz_path = os.path.join(self.artifact_dir, 'model_comparison.png')
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return comparison_df, save_path, viz_path


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("="*50)
    print("Available evaluation functions:")
    print("- calculate_metrics()")
    print("- plot_confusion_matrix()")
    print("- plot_roc_curve()")
    print("- plot_precision_recall_curve()")
    print("- plot_feature_importance()")
    print("- evaluate_model()")
    print("- compare_models()")
    print("- log_to_mlflow()")
