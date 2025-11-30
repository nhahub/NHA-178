# Architecture Documentation

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     PIMA DIABETES ML PIPELINE               │
└─────────────────────────────────────────────────────────────┘

┌──────────────┐
│   Data       │
│   Loading    │──┐
└──────────────┘  │
                  │
┌──────────────┐  │
│  Missing     │  │
│  Values      │◄─┤
│  Imputation  │  │
└──────────────┘  │
                  │
┌──────────────┐  │
│  Feature     │  │
│  Engineering │◄─┤
│  (16 new)    │  │
└──────────────┘  │
                  │
┌──────────────┐  │
│  Train/Test  │  │
│  Split       │◄─┤
│  (Stratified)│  │
└──────────────┘  │
                  │
┌──────────────┐  │
│  Encoding &  │  │
│  Scaling     │◄─┘
└──────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────┐
│         MODEL TRAINING PHASE            │
├─────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐   │
│  │  Logistic    │  │     KNN      │   │
│  │  Regression  │  │              │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │     SVM      │  │   Decision   │   │
│  │              │  │     Tree     │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │   Random     │  │   Gradient   │   │
│  │   Forest     │  │   Boosting   │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐  ┌──────────────┐   │
│  │   XGBoost    │  │  LightGBM    │   │
│  │              │  │              │   │
│  └──────────────┘  └──────────────┘   │
│                                         │
│  ┌──────────────┐                      │
│  │   Neural     │                      │
│  │   Network    │                      │
│  └──────────────┘                      │
└─────────────────────────────────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────┐
│      HYPERPARAMETER TUNING PHASE        │
├─────────────────────────────────────────┤
│  ┌──────────────┐                       │
│  │ GridSearchCV │  (KNN)                │
│  └──────────────┘                       │
│                                         │
│  ┌──────────────┐                       │
│  │RandomizedCV  │  (LightGBM)           │
│  └──────────────┘                       │
│                                         │
│  ┌──────────────┐                       │
│  │    Optuna    │  (RF, XGBoost)        │
│  └──────────────┘                       │
└─────────────────────────────────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────┐
│          ENSEMBLE CREATION              │
├─────────────────────────────────────────┤
│  Voting Classifier (Soft Voting)        │
│  - Combines top performing models       │
│  - Weighted by performance              │
└─────────────────────────────────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────┐
│           EVALUATION PHASE              │
├─────────────────────────────────────────┤
│  • Confusion Matrix                     │
│  • ROC Curve                            │
│  • Precision-Recall Curve               │
│  • Feature Importance                   │
│  • Classification Report                │
│  • Model Comparison                     │
└─────────────────────────────────────────┘
       │
       │
       ▼
┌─────────────────────────────────────────┐
│         MLFLOW TRACKING                 │
├─────────────────────────────────────────┤
│  • Parameters                           │
│  • Metrics                              │
│  • Artifacts (plots, reports)           │
│  • Model Registry                       │
│  • Experiment Tracking                  │
└─────────────────────────────────────────┘
```

## Module Architecture

### 1. preprocess.py
```
PimaPreprocessor
├── load_data()
├── handle_missing_values()
├── feature_engineering()
├── split_data()
├── encode_and_scale()
└── preprocess_pipeline()
```

**Responsibilities:**
- Data loading from Kaggle
- Missing value imputation (target-based)
- 16 feature engineering
- Train/test splitting
- Encoding and scaling

### 2. models.py
```
ModelFactory
├── get_all_models()
├── get_model()
├── get_param_grids()
└── get_param_distributions()
```

**Responsibilities:**
- Model instantiation
- Parameter grid definitions
- Model configuration management

### 3. evaluation.py
```
ModelEvaluator
├── calculate_metrics()
├── plot_confusion_matrix()
├── plot_roc_curve()
├── plot_precision_recall_curve()
├── plot_feature_importance()
├── evaluate_model()
├── compare_models()
└── log_to_mlflow()
```

**Responsibilities:**
- Metric calculation
- Visualization generation
- Artifact creation
- MLflow logging
- Model comparison

### 4. train.py
```
MLflowTrainer
├── train_single_model()
├── train_all_models()
├── hyperparameter_tuning_grid()
├── hyperparameter_tuning_random()
├── hyperparameter_tuning_optuna()
├── create_ensemble()
└── run_complete_pipeline()
```

**Responsibilities:**
- Model training orchestration
- MLflow experiment tracking
- Hyperparameter optimization
- Ensemble creation
- Pipeline execution

### 5. utils.py
```
Utilities
├── timer()
├── save_model() / load_model()
├── save_results() / load_results()
├── create_data_version_info()
├── create_summary_report()
├── ExperimentLogger
└── Visualization utilities
```

**Responsibilities:**
- Helper functions
- Model persistence
- Logging
- Reporting
- Data versioning

## Data Flow

```
Raw Data (diabetes.csv)
    ↓
PimaPreprocessor
    ↓
Cleaned & Engineered Features
    ↓
Train/Test Split
    ↓
Scaled Features
    ↓
ModelFactory → Model Instances
    ↓
MLflowTrainer → Training
    ↓
Trained Models
    ↓
ModelEvaluator → Metrics & Plots
    ↓
MLflow Tracking → Storage
    ↓
Results & Reports
```

## MLflow Integration Points

1. **Experiment Creation**
   - `mlflow.set_experiment()`
   - Organizes all runs

2. **Run Tracking**
   - `mlflow.start_run()`
   - Each model gets unique run

3. **Parameter Logging**
   - `mlflow.log_param()`
   - Model hyperparameters
   - Training configuration

4. **Metric Logging**
   - `mlflow.log_metric()`
   - Accuracy, Precision, Recall, etc.
   - Custom metrics

5. **Artifact Logging**
   - `mlflow.log_artifact()`
   - Plots, reports, models
   - Data versions

6. **Model Registry**
   - `mlflow.sklearn.log_model()`
   - Model versioning
   - Deployment ready

## File Structure Details

```
pima_mlflow_project/
│
├── main.py              # Entry point with CLI
├── predict.py           # Inference script
├── requirements.txt     # Dependencies
├── README.md            # Documentation
├── QUICKSTART.md        # Quick guide
├── .gitignore           # Git exclusions
├── .env.example         # Configuration template
│
├── src/                 # Source code
│   ├── __init__.py
│   ├── preprocess.py   # Data preprocessing
│   ├── models.py       # Model definitions
│   ├── train.py        # Training pipeline
│   ├── evaluation.py   # Evaluation metrics
│   └── utils.py        # Utilities
│
├── data/               # Data storage
│   └── diabetes.csv    # (auto-downloaded)
│
├── models/             # Saved models
│   └── *.pkl          # Pickled models
│
├── mlruns/             # MLflow tracking
│   ├── 0/             # Default experiment
│   └── metadata/      # Experiment metadata
│
├── artifacts/          # Generated artifacts
│   ├── *.png          # Plots
│   ├── *.csv          # Reports
│   └── *.json         # Metadata
│
└── notebooks/          # Optional notebooks
    └── *.ipynb        # Analysis notebooks
```

## Execution Flow

### Standard Execution
```
main.py
  ↓
MLflowTrainer.run_complete_pipeline()
  ↓
1. Data Preprocessing
  ↓
2. Train All Models (Baseline)
  ↓
3. Hyperparameter Tuning (Top 3)
  ↓
4. Create Ensemble
  ↓
5. Evaluation & Comparison
  ↓
6. MLflow Logging
  ↓
7. Summary Report
```

### Quick Execution (No Tuning)
```
main.py --no-tune
  ↓
1. Data Preprocessing
  ↓
2. Train All Models
  ↓
3. Create Ensemble (Top 2)
  ↓
4. Evaluation
  ↓
5. MLflow Logging
```

## Performance Characteristics

- **Memory**: ~2-4 GB for full pipeline
- **CPU**: Utilizes all cores (n_jobs=-1)
- **Disk**: ~500 MB for all artifacts
- **Time**: 
  - Without tuning: 2-5 minutes
  - With tuning: 10-20 minutes

## Extension Points

1. **Add New Models**: Extend `ModelFactory.get_all_models()`
2. **Add Metrics**: Extend `ModelEvaluator.calculate_metrics()`
3. **Add Visualizations**: Add methods to `ModelEvaluator`
4. **Add Tuning Methods**: Add methods to `MLflowTrainer`
5. **Custom Preprocessing**: Extend `PimaPreprocessor`

## Dependencies

### Core
- Python 3.8+
- NumPy, Pandas
- Scikit-learn

### ML Frameworks
- XGBoost
- LightGBM
- Optuna

### Tracking
- MLflow

### Visualization
- Matplotlib
- Seaborn

## Design Patterns

1. **Factory Pattern**: `ModelFactory` for model creation
2. **Pipeline Pattern**: Sequential preprocessing steps
3. **Strategy Pattern**: Different tuning strategies
4. **Observer Pattern**: MLflow callback integration
5. **Singleton Pattern**: ExperimentLogger
