"""
Save the Ensemble Model (LightGBM + KNN) as .pkl file for Streamlit

This script trains and saves the ensemble model from the notebook analysis.
Run this after completing the notebook training.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import kagglehub

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2

print("Loading and preprocessing data...")
# Load data
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
csv_path = os.path.join(path, "diabetes.csv")
data = pd.read_csv(csv_path)

# Replace 0 with NaN for certain features
data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)

# Impute missing values based on target-specific medians
data.loc[(data['Outcome'] == 0) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
data.loc[(data['Outcome'] == 1) & (data['Insulin'].isnull()), 'Insulin'] = 169.5
data.loc[(data['Outcome'] == 0) & (data['Glucose'].isnull()), 'Glucose'] = 107
data.loc[(data['Outcome'] == 1) & (data['Glucose'].isnull()), 'Glucose'] = 140
data.loc[(data['Outcome'] == 0) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27
data.loc[(data['Outcome'] == 1) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32
data.loc[(data['Outcome'] == 0) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70
data.loc[(data['Outcome'] == 1) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
data.loc[(data['Outcome'] == 0) & (data['BMI'].isnull()), 'BMI'] = 30.1
data.loc[(data['Outcome'] == 1) & (data['BMI'].isnull()), 'BMI'] = 34.3

# Feature Engineering (16 features)
print("Creating engineered features...")
data.loc[:,'N1'] = 0
data.loc[(data['Age']<=30) & (data['Glucose']<=120),'N1'] = 1

data.loc[:,'N2'] = 0
data.loc[(data['BMI']<=30),'N2'] = 1

data.loc[:,'N3'] = 0
data.loc[(data['Age']<=30) & (data['Pregnancies']<=6),'N3'] = 1

data.loc[:,'N4'] = 0
data.loc[(data['Glucose']<=105) & (data['BloodPressure']<=80),'N4'] = 1

data.loc[:,'N5'] = 0
data.loc[(data['SkinThickness']<=20),'N5'] = 1

data.loc[:,'N6'] = 0
data.loc[(data['BMI']<30) & (data['SkinThickness']<=20),'N6'] = 1

data.loc[:,'N7'] = 0
data.loc[(data['Glucose']<=105) & (data['BMI']<=30),'N7'] = 1

data.loc[:,'N9'] = 0
data.loc[(data['Insulin']<200),'N9'] = 1

data.loc[:,'N10'] = 0
data.loc[(data['BloodPressure']<80),'N10'] = 1

data.loc[:,'N11'] = 0
data.loc[(data['Pregnancies']<4) & (data['Pregnancies']!=0),'N11'] = 1

data['N0'] = data['BMI'] * data['SkinThickness']
data['N8'] = data['Pregnancies'] / data['Age']
data['N13'] = data['Glucose'] / data['DiabetesPedigreeFunction']
data['N12'] = data['Age'] * data['DiabetesPedigreeFunction']
data['N14'] = data['Age'] / data['Insulin']

data.loc[:,'N15'] = 0
data.loc[(data['N0']<1034),'N15'] = 1

# Split data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Preprocessing
print("Preprocessing data...")
cat_cols = X_train.nunique()[X_train.nunique() < 12].keys().tolist()
num_cols = [x for x in X_train.columns if x not in cat_cols]
bin_cols = X_train.nunique()[X_train.nunique() == 2].keys().tolist()
multi_cols = [i for i in cat_cols if i not in bin_cols]

X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# Label encoding for binary columns
label_encoders = {}
for col in bin_cols:
    le = LabelEncoder()
    X_train_processed[col] = le.fit_transform(X_train_processed[col])
    X_test_processed[col] = le.transform(X_test_processed[col])
    label_encoders[col] = le

# One-hot encoding for multi-class columns
if multi_cols:
    X_train_processed = pd.get_dummies(X_train_processed, columns=multi_cols, prefix=multi_cols)
    X_test_processed = pd.get_dummies(X_test_processed, columns=multi_cols, prefix=multi_cols)
    
    train_cols = set(X_train_processed.columns)
    test_cols = set(X_test_processed.columns)
    
    for col in train_cols - test_cols:
        X_test_processed[col] = 0
    
    for col in test_cols - train_cols:
        X_test_processed = X_test_processed.drop(col, axis=1)
    
    X_test_processed = X_test_processed[X_train_processed.columns]

# Standard scaling
scaler = StandardScaler()
X_train_scaled = X_train_processed.copy()
X_test_scaled = X_test_processed.copy()

if num_cols:
    scaler.fit(X_train_processed[num_cols])
    X_train_scaled[num_cols] = scaler.transform(X_train_processed[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_processed[num_cols])

# Train optimized ensemble model
print("Training ensemble model (LightGBM + KNN)...")

# Best parameters from notebook optimization
lgbm_optimized = LGBMClassifier(
    random_state=RANDOM_STATE,
    verbose=-1,
    learning_rate=0.1,
    n_estimators=200,
    num_leaves=20,
    max_depth=5
)

knn_optimized = KNeighborsClassifier(
    n_neighbors=11,
    weights='distance',
    metric='euclidean'
)

# Create ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_optimized),
        ('knn', knn_optimized)
    ],
    voting='soft',
    weights=[1, 1]
)

voting_clf.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = voting_clf.predict(X_test_scaled)
y_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\nEnsemble Model Performance:")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Save models
print("\nSaving models...")
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Save ensemble model
ensemble_model_path = os.path.join(models_dir, 'ensemble_lgbm_knn.pkl')
with open(ensemble_model_path, 'wb') as f:
    pickle.dump(voting_clf, f)

# Save scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names
feature_names_path = os.path.join(models_dir, 'feature_names.pkl')
with open(feature_names_path, 'wb') as f:
    pickle.dump(list(X_train_scaled.columns), f)

# Save label encoders
encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
with open(encoders_path, 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✅ All files saved successfully!")
print("="*50)
print(f"1. Ensemble Model: {ensemble_model_path}")
print(f"   Size: {os.path.getsize(ensemble_model_path) / 1024:.2f} KB")
print(f"2. Scaler: {scaler_path}")
print(f"3. Feature Names: {feature_names_path}")
print(f"4. Label Encoders: {encoders_path}")
print("\nThese files are ready to use in your Streamlit app!")
