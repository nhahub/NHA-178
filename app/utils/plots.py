"""
Plotting Utilities for Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import streamlit as st

def plot_confusion_matrix(cm_or_y_test, y_pred=None, title="Confusion Matrix"):
    """
    Plot confusion matrix (supports both confusion matrix and y_test/y_pred)
    
    Args:
        cm_or_y_test: Either confusion matrix or true labels
        y_pred: Predicted labels (if first arg is y_test)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    # Check if first argument is confusion matrix or y_test
    if y_pred is None:
        # First argument is confusion matrix
        cm = cm_or_y_test
    else:
        # First argument is y_test, compute confusion matrix
        cm = confusion_matrix(cm_or_y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_roc_curve(y_test, y_proba, title="ROC Curve"):
    """
    Plot ROC curve
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_precision_recall_curve(y_test, y_proba, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='red', lw=2, 
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance_horizontal(model, feature_names, top_n=10, title="Feature Importance"):
    """
    Plot horizontal feature importance
    
    Args:
        model: Trained model with feature_importances_ OR feature_names list with importances as second arg
        feature_names: List of feature names OR importances array
        top_n: Number of top features to show
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    # Handle both function signatures
    if isinstance(model, (list, np.ndarray)):
        # Called with (feature_names, importances)
        feature_names_list = model
        importances = feature_names
    else:
        # Called with (model, feature_names)
        if not hasattr(model, 'feature_importances_'):
            return None
        importances = model.feature_importances_
        feature_names_list = feature_names
    
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(indices))
    
    ax.barh(y_pos, importances[indices], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names_list[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance_vertical(model, feature_names, top_n=10, title="Feature Importance"):
    """
    Plot vertical feature importance
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(indices))
    
    ax.bar(x_pos, importances[indices], color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_distribution_comparison(data, column, title="Distribution"):
    """
    Plot distribution for a single column
    
    Args:
        data: DataFrame
        column: Column name
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if data[column].dtype in ['int64', 'float64']:
        ax.hist(data[column].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    else:
        value_counts = data[column].value_counts()
        ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(data, title="Correlation Matrix"):
    """
    Plot correlation heatmap
    
    Args:
        data: DataFrame
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_prediction_distribution(predictions, probabilities, title="Prediction Distribution"):
    """
    Plot prediction distribution
    
    Args:
        predictions: Array of predictions
        probabilities: Array of probabilities
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prediction counts
    unique, counts = np.unique(predictions, return_counts=True)
    colors = ['#2ECC71' if x == 0 else '#E74C3C' for x in unique]
    ax1.bar(['No Diabetes', 'Diabetes'] if len(unique) == 2 else [str(x) for x in unique], 
            counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('Prediction Counts', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Probability distribution
    ax2.hist(probabilities, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.set_xlabel('Probability of Diabetes', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_dict, title="Metrics Comparison"):
    """
    Plot bar chart comparing different metrics
    
    Args:
        metrics_dict: Dictionary with metric names and values
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics)))
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
