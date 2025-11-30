"""
Model Utilities for Training and Preprocessing
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def get_model(model_type, random_state=42, custom_params=None):
    """
    Get ML model instance
    
    Args:
        model_type: Name of the model
        random_state: Random seed
        custom_params: Custom hyperparameters
    
    Returns:
        Model instance
    """
    params = custom_params if custom_params else {}
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=-1, **params),
        "XGBoost": XGBClassifier(random_state=random_state, eval_metric='logloss', n_jobs=-1, **params),
        "LightGBM": LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1, **params),
        "Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000, **params),
        "SVM": SVC(random_state=random_state, probability=True, **params),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state, **params),
        "KNN": KNeighborsClassifier(n_jobs=-1, **params),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state, **params),
    }
    
    return models.get(model_type)

def preprocess_data(data, test_size=0.2, random_state=42, apply_feature_engineering=True):
    """
    Preprocess dataset for training (MATCHES NOTEBOOK)
    
    Args:
        data: Raw dataframe
        test_size: Test split ratio
        random_state: Random seed
        apply_feature_engineering: Whether to apply feature engineering (16 features)
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Separate features and target
    if 'Outcome' not in data.columns:
        raise ValueError("Dataset must contain 'Outcome' column")
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Step 1: Handle missing values (replace 0 with NaN for medical impossibilities)
    df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
        df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
    
    # Step 2: Impute missing values using target-specific medians (from notebook)
    # Insulin
    df.loc[(df['Outcome'] == 0) & (df['Insulin'].isnull()), 'Insulin'] = 102.5
    df.loc[(df['Outcome'] == 1) & (df['Insulin'].isnull()), 'Insulin'] = 169.5
    
    # Glucose
    df.loc[(df['Outcome'] == 0) & (df['Glucose'].isnull()), 'Glucose'] = 107
    df.loc[(df['Outcome'] == 1) & (df['Glucose'].isnull()), 'Glucose'] = 140
    
    # SkinThickness
    df.loc[(df['Outcome'] == 0) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27
    df.loc[(df['Outcome'] == 1) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 32
    
    # BloodPressure
    df.loc[(df['Outcome'] == 0) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70
    df.loc[(df['Outcome'] == 1) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
    
    # BMI
    df.loc[(df['Outcome'] == 0) & (df['BMI'].isnull()), 'BMI'] = 30.1
    df.loc[(df['Outcome'] == 1) & (df['BMI'].isnull()), 'BMI'] = 34.3
    
    # Step 3: Feature Engineering (16 new features from notebook)
    if apply_feature_engineering:
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
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Step 4: Train-test split (BEFORE scaling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Step 5: Get column types
    cat_cols = X_train.nunique()[X_train.nunique() < 12].keys().tolist()
    num_cols = [x for x in X_train.columns if x not in cat_cols]
    bin_cols = X_train.nunique()[X_train.nunique() == 2].keys().tolist()
    multi_cols = [i for i in cat_cols if i not in bin_cols]
    
    # Create copies
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Step 6: Label encoding for binary columns
    label_encoders = {}
    for col in bin_cols:
        le = LabelEncoder()
        X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
        X_test_processed[col] = le.transform(X_test_processed[col].astype(str))
        label_encoders[col] = le
    
    # Step 7: One-hot encoding for multi-class columns
    if multi_cols:
        X_train_processed = pd.get_dummies(X_train_processed, columns=multi_cols, prefix=multi_cols)
        X_test_processed = pd.get_dummies(X_test_processed, columns=multi_cols, prefix=multi_cols)
        
        # Align columns
        train_cols = set(X_train_processed.columns)
        test_cols = set(X_test_processed.columns)
        
        for col in train_cols - test_cols:
            X_test_processed[col] = 0
        
        for col in test_cols - train_cols:
            X_test_processed = X_test_processed.drop(col, axis=1)
        
        X_test_processed = X_test_processed[X_train_processed.columns]
    
    # Step 8: Standard scaling for numerical columns (fit on train, transform both)
    if num_cols:
        scaler = StandardScaler()
        X_train_processed[num_cols] = scaler.fit_transform(X_train_processed[num_cols])
        X_test_processed[num_cols] = scaler.transform(X_test_processed[num_cols])
    
    feature_names = X_train_processed.columns.tolist()
    
    return X_train_processed, X_test_processed, y_train, y_test, feature_names

def train_model_with_mlflow(model_type, X_train, X_test, y_train, y_test, 
                            feature_names, experiment_name, custom_params=None, random_state=42):
    """
    Train model and log to MLflow
    
    Args:
        model_type: Name of the model
        X_train, X_test, y_train, y_test: Training and test data
        feature_names: List of feature names
        experiment_name: MLflow experiment name
        custom_params: Custom hyperparameters
        random_state: Random seed
    
    Returns:
        model, metrics, run_id
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Get model
        model = get_model(model_type, random_state, custom_params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = None
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        if y_test_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_test_proba)
        
        # Log parameters
        mlflow.log_param("model_name", model_type)
        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        
        # Log custom params
        if custom_params:
            for key, value in custom_params.items():
                mlflow.log_param(key, value)
        
        # Log metrics
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                mlflow.log_metric(key, value)
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{model_type.replace(' ', '_')}")
        
        # Get run ID
        run_id = run.info.run_id
    
    return model, metrics, run_id

def create_ensemble_model(X_train, X_test, y_train, y_test, 
                         feature_names, experiment_name, random_state=42):
    """
    Create and train ensemble model (same as notebook)
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        feature_names: List of feature names
        experiment_name: MLflow experiment name
        random_state: Random seed
    
    Returns:
        model, metrics, run_id
    """
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Create base models (LightGBM + KNN as in notebook)
        lgbm = LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=-1)
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('lgbm', lgbm),
                ('knn', knn)
            ],
            voting='soft',
            weights=[1, 1]
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = ensemble.predict(X_train)
        y_test_pred = ensemble.predict(X_test)
        y_test_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        # Log parameters
        mlflow.log_param("model_name", "Ensemble (LightGBM + KNN)")
        mlflow.log_param("model_type", "VotingClassifier")
        mlflow.log_param("base_models", "LightGBM, KNN")
        mlflow.log_param("voting", "soft")
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_test_samples", X_test.shape[0])
        
        # Log metrics
        for key, value in metrics.items():
            if key != 'confusion_matrix':
                mlflow.log_metric(key, value)
        
        # Log model
        mlflow.sklearn.log_model(ensemble, "model_ensemble")
        
        # Get run ID
        run_id = run.info.run_id
    
    return ensemble, metrics, run_id
