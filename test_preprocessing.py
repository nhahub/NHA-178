"""
Test Script to Verify Preprocessing Matches Notebook
Run this to confirm 87% accuracy is achievable
"""

import pandas as pd
import numpy as np
import os
import kagglehub
from app.utils.model_utils import preprocess_data, train_model_with_mlflow, create_ensemble_model

print("="*60)
print("TESTING STREAMLIT PREPROCESSING")
print("="*60)

# Load data
print("\n1. Loading dataset...")
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
csv_path = os.path.join(path, "diabetes.csv")
data = pd.read_csv(csv_path)
print(f"   Dataset shape: {data.shape}")

# Preprocess
print("\n2. Preprocessing data (with feature engineering)...")
X_train, X_test, y_train, y_test, feature_names = preprocess_data(
    data, test_size=0.2, random_state=42, apply_feature_engineering=True
)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")
print(f"   Total features: {len(feature_names)}")
print(f"   Features: {feature_names[:10]}... (showing first 10)")

# Check for engineered features
engineered_features = [f for f in feature_names if f.startswith('N')]
print(f"\n3. Engineered features found: {len(engineered_features)}")
print(f"   {engineered_features}")

# Train ensemble model
print("\n4. Training Ensemble Model (LightGBM + KNN)...")
model, metrics, run_id = create_ensemble_model(
    X_train, X_test, y_train, y_test,
    feature_names, "Test_Ensemble", random_state=42
)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Test Accuracy:  {metrics['test_accuracy']:.4f} (target: ~0.87)")
print(f"Train Accuracy: {metrics['train_accuracy']:.4f}")
print(f"ROC-AUC:        {metrics['roc_auc']:.4f} (target: ~0.93)")
print(f"Precision:      {metrics['test_precision']:.4f}")
print(f"Recall:         {metrics['test_recall']:.4f}")
print(f"F1-Score:       {metrics['test_f1']:.4f} (target: ~0.82)")
print(f"Overfitting:    {metrics['train_accuracy'] - metrics['test_accuracy']:.4f}")
print("="*60)

if metrics['test_accuracy'] >= 0.85:
    print("\n✅ SUCCESS! Accuracy matches notebook performance!")
else:
    print("\n⚠️  Accuracy lower than expected. Check preprocessing pipeline.")

print(f"\n📁 MLflow Run ID: {run_id}")
print("\nTest complete! You can now use the Streamlit app with confidence.")
