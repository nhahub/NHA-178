# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

### Option 1: Using main.py
```bash
python main.py
```

### Option 2: Using train.py directly
```bash
cd src
python train.py
```

### Option 3: Import as module
```python
from src.train import MLflowTrainer

trainer = MLflowTrainer()
results = trainer.run_complete_pipeline()
```

## View Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser to:
# http://localhost:5000
```

## Command-Line Options

```bash
# Disable hyperparameter tuning (faster)
python main.py --no-tune

# Custom experiment name
python main.py --experiment-name "My_Experiment"

# Custom random state
python main.py --random-state 123

# Use custom dataset
python main.py --csv-path path/to/diabetes.csv
```

## Expected Execution Time

- Without tuning: ~2-5 minutes
- With tuning: ~10-20 minutes

## Output Files

- `mlruns/`: MLflow tracking data
- `artifacts/`: Plots and visualizations
- `training.log`: Execution log
- `model_summary_report.txt`: Results summary

## Troubleshooting

### Issue: Import errors
```bash
# Ensure you're in project root and src is accessible
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%\src  # Windows
```

### Issue: Dataset download fails
- Ensure internet connection
- Or manually download diabetes.csv to data/ folder

### Issue: MLflow UI won't start
```bash
# Use different port
mlflow ui --port 5001
```

## Next Steps

1. ✅ Run the pipeline
2. ✅ Explore MLflow UI
3. ✅ Check artifacts/
4. ✅ Read model_summary_report.txt
5. ✅ Experiment with hyperparameters
6. ✅ Deploy best model
