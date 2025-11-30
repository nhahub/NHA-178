# 🚀 Streamlit Web Application Guide

## Overview

This guide will help you set up and run the **Pima Indians Diabetes Classification Streamlit Web Application**. The app provides an interactive interface for exploring the dataset, training ML models, making predictions, and browsing MLflow experiments.

---

## 📋 Prerequisites

- **Python 3.8+** installed on your system
- **pip** package manager
- **Git** (optional, for cloning the repository)

---

## 🛠️ Installation Steps

### 1. Navigate to Project Directory

```bash
cd pima_mlflow_project
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- `streamlit` - Web application framework
- `mlflow` - Experiment tracking and model registry
- `scikit-learn`, `xgboost`, `lightgbm` - ML algorithms
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- And more...

---

## 🎯 Running the Application

### Start the Streamlit Server

```bash
streamlit run streamlit_app.py
```

### Access the Application

Once the server starts, you'll see output like:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.X:8501
```

Open your browser and navigate to:
```
http://localhost:8501
```

---

## 📁 Application Structure

```
pima_mlflow_project/
│
├── streamlit_app.py          # Main application entry point
│
├── app/                       # Application package
│   ├── __init__.py
│   │
│   ├── pages/                 # Page modules
│   │   ├── __init__.py
│   │   ├── home.py            # Home dashboard
│   │   ├── dataset_explorer.py  # Dataset EDA
│   │   ├── training.py        # Model training
│   │   ├── predict.py         # Predictions
│   │   └── model_explorer.py  # MLflow browser
│   │
│   └── utils/                 # Utility modules
│       ├── __init__.py
│       ├── model_utils.py     # Model training functions
│       ├── mlflow_utils.py    # MLflow operations
│       └── plots.py           # Plotting functions
│
├── data/                      # Dataset directory
├── models/                    # Saved models
├── mlruns/                    # MLflow tracking data
├── artifacts/                 # Model artifacts
└── requirements.txt           # Python dependencies
```

---

## 🎨 Application Features

### 1. **🏠 Home Dashboard**
- Project overview and introduction
- Quick statistics (9 algorithms, 24 features, 87.66% accuracy)
- Key metrics display (ROC-AUC: 93.19%)
- Technology stack showcase
- Navigation to other pages

### 2. **📊 Dataset Explorer**
- Upload custom CSV datasets or use default Pima dataset
- **5 Interactive Tabs:**
  - **Overview**: Data preview, shape, column types
  - **Statistics**: Descriptive statistics for all features
  - **Distributions**: Histograms and distribution plots
  - **Correlations**: Correlation heatmap
  - **Missing Values**: Missing data analysis and visualization

### 3. **🎓 Model Training**
- **9 ML Algorithms:** Random Forest, XGBoost, LightGBM, Logistic Regression, SVM, Gradient Boosting, KNN, Decision Tree, Ensemble
- Custom hyperparameter tuning with interactive sliders
- Real-time training progress indicator
- Performance metrics display (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix and ROC curve visualization
- MLflow integration (automatic logging of parameters, metrics, and artifacts)
- Download trained models as `.pkl` files

### 4. **🔮 Predictions**
- **Two Input Methods:**
  - **Manual Input**: Sliders for 8 key features (Pregnancies, Age, Glucose, Insulin, BMI, Blood Pressure, Skin Thickness, DiabetesPedigreeFunction)
  - **CSV Upload**: Batch predictions for multiple samples
- Color-coded prediction results (Green = Healthy, Red = Diabetic)
- Probability display with progress bars
- Feature importance chart (for tree-based models)
- Download prediction results as CSV

### 5. **📈 MLflow Model Explorer**
- Connect to MLflow tracking server
- Browse all experiments
- View runs with metrics table
- **4-Tab Run Details:**
  - **Metrics**: Key performance indicators
  - **Parameters**: Model hyperparameters
  - **Artifacts**: Model files and visualizations
  - **Actions**: Activate or delete runs
- Compare multiple runs
- Set active model for predictions

---

## 🚦 Quick Start Workflow

### Step 1: Launch Application
```bash
streamlit run streamlit_app.py
```

### Step 2: Explore Dataset
1. Navigate to **📊 Dataset Explorer**
2. Upload your CSV or use the default Pima dataset
3. Explore data through 5 interactive tabs

### Step 3: Train a Model
1. Go to **🎓 Model Training**
2. Upload training dataset
3. Select a model (e.g., Random Forest)
4. Adjust hyperparameters (optional)
5. Click **🚀 Train Model**
6. View results and download model

### Step 4: Make Predictions
1. Navigate to **🔮 Predictions**
2. Upload a saved model (`.pkl` file)
3. Choose input method:
   - **Manual**: Use sliders to input values
   - **CSV**: Upload a file with test samples
4. Click **Predict** and view results

### Step 5: Browse MLflow Experiments
1. Go to **📈 MLflow Model Explorer**
2. Connect to MLflow (auto-configured)
3. Browse experiments and runs
4. View metrics, parameters, and artifacts
5. Set active model for predictions

---

## 🐛 Troubleshooting

### Issue: Port Already in Use

**Error:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Issue: Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: MLflow Tracking URI Error

**Error:**
```
MlflowException: Could not connect to tracking server
```

**Solution:**
- Ensure `mlruns/` directory exists in project root
- Check MLflow URI in Model Explorer page
- Verify MLflow is installed: `pip show mlflow`

### Issue: Dataset Not Found

**Error:**
```
FileNotFoundError: data/diabetes.csv not found
```

**Solution:**
- Run `python main.py` first to download dataset
- Or upload your own dataset in Dataset Explorer

---

## 📝 Configuration Options

### Streamlit Configuration (Optional)

Create `.streamlit/config.toml`:

```toml
[server]
port = 8501
maxUploadSize = 200

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### MLflow Configuration

The app automatically uses the local `mlruns/` directory. To use a remote tracking server:

1. Set environment variable:
```bash
export MLFLOW_TRACKING_URI=http://your-server:5000
```

2. Or configure in Model Explorer page

---

## 🔧 Advanced Usage

### Running MLflow UI Alongside Streamlit

**Terminal 1 (Streamlit):**
```bash
streamlit run streamlit_app.py
```

**Terminal 2 (MLflow UI):**
```bash
mlflow ui --port 5000
```

Access:
- Streamlit App: `http://localhost:8501`
- MLflow UI: `http://localhost:5000`

### Custom Dataset Format

For predictions, your CSV should have these columns:
- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Project README**: See `README.md` for full project details

---

## 👤 Author

**Hossam Medhat**  
📧 hossammedhat81@gmail.com  
🔗 GitHub: [Your GitHub Profile]

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 🎉 Enjoy Using the Application!

If you encounter any issues or have suggestions, please open an issue on GitHub or contact the author.

**Happy Modeling! 🚀**
