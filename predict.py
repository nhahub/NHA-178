"""
Prediction Script - Make predictions using trained models from MLflow
"""

import os
import sys
import argparse
import pandas as pd
import mlflow.sklearn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocess import PimaPreprocessor


def predict_from_model(model_uri, data_path=None, sample_data=None):
    """Make predictions using a trained model from MLflow"""
    
    # Load model
    print(f"Loading model from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    
    # Prepare data
    if data_path:
        # Load from file
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
    elif sample_data is not None:
        # Use provided sample data
        data = pd.DataFrame([sample_data])
    else:
        # Use sample input
        print("Using sample data for prediction...")
        sample_data = {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        }
        data = pd.DataFrame([sample_data])
    
    print(f"\nInput data shape: {data.shape}")
    print(f"\nInput features:")
    print(data)
    
    # Note: In production, you would need to apply the same preprocessing
    # as during training. For now, we assume the model expects raw features.
    
    # Make predictions
    predictions = model.predict(data)
    probabilities = model.predict_proba(data) if hasattr(model, 'predict_proba') else None
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, pred in enumerate(predictions):
        result = "Diabetic" if pred == 1 else "Not Diabetic"
        print(f"\nSample {i+1}: {result}")
        
        if probabilities is not None:
            prob_no_diabetes = probabilities[i][0] * 100
            prob_diabetes = probabilities[i][1] * 100
            print(f"  Probability (Not Diabetic): {prob_no_diabetes:.2f}%")
            print(f"  Probability (Diabetic): {prob_diabetes:.2f}%")
    
    print("="*60)
    
    return predictions, probabilities


def main():
    parser = argparse.ArgumentParser(description="Make predictions using trained models")
    
    parser.add_argument(
        '--model-uri',
        type=str,
        required=True,
        help='MLflow model URI (e.g., runs:/<run_id>/model_LightGBM or models:/ModelName/1)'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to CSV file with input data'
    )
    
    args = parser.parse_args()
    
    # Make predictions
    predictions, probabilities = predict_from_model(
        model_uri=args.model_uri,
        data_path=args.data_path
    )
    
    return 0


if __name__ == "__main__":
    main()
