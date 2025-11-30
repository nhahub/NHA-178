"""
Data Preprocessing Module for Pima Indians Diabetes Dataset
Handles data loading, cleaning, imputation, feature engineering, and scaling
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub


class PimaPreprocessor:
    """Complete preprocessing pipeline for Pima dataset"""
    
    def __init__(self, random_state=42, test_size=0.2):
        self.random_state = random_state
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, csv_path=None):
        """Load Pima Indians Diabetes dataset"""
        if csv_path is None:
            print("Downloading Pima Indians Diabetes dataset from Kaggle...")
            path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
            csv_path = os.path.join(path, "diabetes.csv")
        
        print(f"Loading dataset from: {csv_path}")
        data = pd.read_csv(csv_path)
        print(f"Dataset loaded successfully! Shape: {data.shape}")
        return data
    
    def handle_missing_values(self, data):
        """Handle missing values (zeros that represent missing data)"""
        print("Handling missing values...")
        
        # Replace 0 with NaN for features where 0 is medically impossible
        cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
        
        # Target-based median imputation
        def median_target(var):
            temp = data[data[var].notnull()]
            temp = temp[[var, 'Outcome']].groupby(['Outcome'])[[var]].median().reset_index()
            return temp
        
        # Insulin
        data.loc[(data['Outcome'] == 0) & (data['Insulin'].isnull()), 'Insulin'] = 102.5
        data.loc[(data['Outcome'] == 1) & (data['Insulin'].isnull()), 'Insulin'] = 169.5
        
        # Glucose
        data.loc[(data['Outcome'] == 0) & (data['Glucose'].isnull()), 'Glucose'] = 107
        data.loc[(data['Outcome'] == 1) & (data['Glucose'].isnull()), 'Glucose'] = 140
        
        # SkinThickness
        data.loc[(data['Outcome'] == 0) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 27
        data.loc[(data['Outcome'] == 1) & (data['SkinThickness'].isnull()), 'SkinThickness'] = 32
        
        # BloodPressure
        data.loc[(data['Outcome'] == 0) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 70
        data.loc[(data['Outcome'] == 1) & (data['BloodPressure'].isnull()), 'BloodPressure'] = 74.5
        
        # BMI
        data.loc[(data['Outcome'] == 0) & (data['BMI'].isnull()), 'BMI'] = 30.1
        data.loc[(data['Outcome'] == 1) & (data['BMI'].isnull()), 'BMI'] = 34.3
        
        print(f"Missing values after imputation: {data.isnull().sum().sum()}")
        return data
    
    def feature_engineering(self, data):
        """Create 16 engineered features based on domain knowledge"""
        print("Creating 16 engineered features...")
        
        # Binary features based on thresholds
        data.loc[:, 'N1'] = 0
        data.loc[(data['Age'] <= 30) & (data['Glucose'] <= 120), 'N1'] = 1
        
        data.loc[:, 'N2'] = 0
        data.loc[(data['BMI'] <= 30), 'N2'] = 1
        
        data.loc[:, 'N3'] = 0
        data.loc[(data['Age'] <= 30) & (data['Pregnancies'] <= 6), 'N3'] = 1
        
        data.loc[:, 'N4'] = 0
        data.loc[(data['Glucose'] <= 105) & (data['BloodPressure'] <= 80), 'N4'] = 1
        
        data.loc[:, 'N5'] = 0
        data.loc[(data['SkinThickness'] <= 20), 'N5'] = 1
        
        data.loc[:, 'N6'] = 0
        data.loc[(data['BMI'] < 30) & (data['SkinThickness'] <= 20), 'N6'] = 1
        
        data.loc[:, 'N7'] = 0
        data.loc[(data['Glucose'] <= 105) & (data['BMI'] <= 30), 'N7'] = 1
        
        data.loc[:, 'N9'] = 0
        data.loc[(data['Insulin'] < 200), 'N9'] = 1
        
        data.loc[:, 'N10'] = 0
        data.loc[(data['BloodPressure'] < 80), 'N10'] = 1
        
        data.loc[:, 'N11'] = 0
        data.loc[(data['Pregnancies'] < 4) & (data['Pregnancies'] != 0), 'N11'] = 1
        
        # Continuous features
        data['N0'] = data['BMI'] * data['SkinThickness']
        data['N8'] = data['Pregnancies'] / data['Age']
        data['N13'] = data['Glucose'] / data['DiabetesPedigreeFunction']
        data['N12'] = data['Age'] * data['DiabetesPedigreeFunction']
        data['N14'] = data['Age'] / data['Insulin']
        
        # Additional feature from N0
        data.loc[:, 'N15'] = 0
        data.loc[(data['N0'] < 1034), 'N15'] = 1
        
        print(f"Feature engineering completed. New shape: {data.shape}")
        return data
    
    def split_data(self, data):
        """Split data into train and test sets"""
        print("Splitting data into train and test sets...")
        
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Training target distribution:\n{y_train.value_counts()}")
        print(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def encode_and_scale(self, X_train, X_test):
        """Encode categorical features and scale numerical features"""
        print("Encoding and scaling features...")
        
        # Identify column types
        cat_cols = X_train.nunique()[X_train.nunique() < 12].keys().tolist()
        num_cols = [x for x in X_train.columns if x not in cat_cols]
        bin_cols = X_train.nunique()[X_train.nunique() == 2].keys().tolist()
        multi_cols = [i for i in cat_cols if i not in bin_cols]
        
        print(f"Numerical columns: {len(num_cols)}")
        print(f"Binary columns: {len(bin_cols)}")
        print(f"Multi-class categorical columns: {len(multi_cols)}")
        
        # Create copies
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()
        
        # Label encoding for binary columns
        for col in bin_cols:
            le = LabelEncoder()
            X_train_processed[col] = le.fit_transform(X_train_processed[col])
            X_test_processed[col] = le.transform(X_test_processed[col])
            self.label_encoders[col] = le
        
        # One-hot encoding for multi-class columns
        if multi_cols:
            X_train_processed = pd.get_dummies(X_train_processed, columns=multi_cols, prefix=multi_cols)
            X_test_processed = pd.get_dummies(X_test_processed, columns=multi_cols, prefix=multi_cols)
            
            # Ensure same columns
            train_cols = set(X_train_processed.columns)
            test_cols = set(X_test_processed.columns)
            
            for col in train_cols - test_cols:
                X_test_processed[col] = 0
            
            for col in test_cols - train_cols:
                X_test_processed = X_test_processed.drop(col, axis=1)
            
            X_test_processed = X_test_processed[X_train_processed.columns]
        
        # Standard scaling for numerical columns
        X_train_scaled = X_train_processed.copy()
        X_test_scaled = X_test_processed.copy()
        
        if num_cols:
            self.scaler.fit(X_train_processed[num_cols])
            X_train_scaled[num_cols] = self.scaler.transform(X_train_processed[num_cols])
            X_test_scaled[num_cols] = self.scaler.transform(X_test_processed[num_cols])
        
        self.feature_names = X_train_scaled.columns.tolist()
        
        print(f"Preprocessing completed!")
        print(f"Final training shape: {X_train_scaled.shape}")
        print(f"Final test shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def preprocess_pipeline(self, csv_path=None):
        """Complete preprocessing pipeline"""
        print("="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load data
        data = self.load_data(csv_path)
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Feature engineering
        data = self.feature_engineering(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(data)
        
        # Encode and scale
        X_train_scaled, X_test_scaled = self.encode_and_scale(X_train, X_test)
        
        print("="*60)
        print("PREPROCESSING PIPELINE COMPLETED")
        print("="*60)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, data


def get_preprocessed_data(csv_path=None, random_state=42, test_size=0.2):
    """Convenience function to get preprocessed data"""
    preprocessor = PimaPreprocessor(random_state=random_state, test_size=test_size)
    return preprocessor.preprocess_pipeline(csv_path)


if __name__ == "__main__":
    # Test preprocessing pipeline
    X_train, X_test, y_train, y_test, data = get_preprocessed_data()
    print(f"\nFinal dataset summary:")
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Feature names: {X_train.columns.tolist()[:10]}...")
