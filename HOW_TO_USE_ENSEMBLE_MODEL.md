# How to Use the Ensemble Model in Streamlit

## ✅ Model Files Created

The following files have been saved in the `models/` directory:

1. **ensemble_lgbm_knn.pkl** (449 KB) - The trained ensemble model (LightGBM + KNN)
2. **scaler.pkl** - StandardScaler for preprocessing numerical features
3. **feature_names.pkl** - List of feature names expected by the model
4. **label_encoders.pkl** - Label encoders for binary categorical features

## 📊 Model Performance

- **Test Accuracy:** 88.96%
- **ROC AUC:** 91.89%

## 🔧 How to Load and Use in Streamlit

### Option 1: Direct Upload in Predictions Page

1. Go to your Streamlit app's **Predictions** page
2. Use the file uploader to upload `ensemble_lgbm_knn.pkl`
3. The model will be loaded automatically

### Option 2: Load Programmatically in Code

```python
import pickle

# Load the ensemble model
with open('models/ensemble_lgbm_knn.pkl', 'rb') as f:
    ensemble_model = pickle.load(f)

# Load the scaler (if needed for preprocessing)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load label encoders
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Make predictions
predictions = ensemble_model.predict(X_preprocessed)
probabilities = ensemble_model.predict_proba(X_preprocessed)
```

### Option 3: Use in Training Page

In your `app/pages/training.py`, you can add the ensemble model to the model selection:

```python
model_type = st.selectbox(
    "Select Model Type",
    ["Logistic Regression", "Random Forest", "SVM", "Decision Tree", 
     "XGBoost", "Gradient Boosting", "LightGBM", "KNN", 
     "Ensemble (LightGBM + KNN)"]  # Add this option
)

if model_type == "Ensemble (LightGBM + KNN)":
    # Load pre-trained ensemble
    with open('models/ensemble_lgbm_knn.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Loaded pre-trained ensemble model!")
```

## 🎯 Expected Input Features

The model expects **24 features** in this exact order:

### Original Features (8):
1. Pregnancies
2. Glucose
3. BloodPressure
4. SkinThickness
5. Insulin
6. BMI
7. DiabetesPedigreeFunction
8. Age

### Engineered Features (16):
9. N0 - BMI × SkinThickness
10. N1 - Binary: (Age ≤ 30) & (Glucose ≤ 120)
11. N2 - Binary: BMI ≤ 30
12. N3 - Binary: (Age ≤ 30) & (Pregnancies ≤ 6)
13. N4 - Binary: (Glucose ≤ 105) & (BloodPressure ≤ 80)
14. N5 - Binary: SkinThickness ≤ 20
15. N6 - Binary: (BMI < 30) & (SkinThickness ≤ 20)
16. N7 - Binary: (Glucose ≤ 105) & (BMI ≤ 30)
17. N8 - Pregnancies / Age
18. N9 - Binary: Insulin < 200
19. N10 - Binary: BloodPressure < 80
20. N11 - Binary: (Pregnancies < 4) & (Pregnancies ≠ 0)
21. N12 - Age × DiabetesPedigreeFunction
22. N13 - Glucose / DiabetesPedigreeFunction
23. N14 - Age / Insulin
24. N15 - Binary: N0 < 1034

## 📝 Important Notes

1. **Preprocessing Required:** The input data must be preprocessed exactly as in the notebook:
   - Missing value imputation (target-specific medians)
   - Feature engineering (16 features)
   - Label encoding for binary features
   - Standard scaling for numerical features

2. **Compatibility:** The model uses the preprocessing pipeline from your `app/utils/model_utils.py` which already implements this logic.

3. **Model Type:** This is a VotingClassifier with:
   - LightGBM (optimized hyperparameters)
   - KNN (optimized hyperparameters)
   - Soft voting with equal weights

## 🚀 Quick Test

To verify the model works:

```python
import pickle
import numpy as np

# Load model
with open('models/ensemble_lgbm_knn.pkl', 'rb') as f:
    model = pickle.load(f)

# Test with sample data (must have 24 features)
sample = np.random.rand(1, 24)  # Replace with actual preprocessed data
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

## ✅ Verification

The model has been tested and achieves:
- **88.96% accuracy** on test data
- **91.89% ROC AUC** score

This matches the expected performance from the notebook analysis!
