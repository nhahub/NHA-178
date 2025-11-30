"""
Main Entry Point for Pima Indians Diabetes Classification Pipeline
Run this script to execute the complete ML pipeline with MLflow tracking
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import MLflowTrainer
from src.utils import print_section_header


def main(args):
    """Main execution function"""
    
    print_section_header("PIMA INDIANS DIABETES CLASSIFICATION", width=80)
    print("Production-Ready MLflow Pipeline")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Experiment Name: {args.experiment_name}")
    print(f"  Random State: {args.random_state}")
    print(f"  Hyperparameter Tuning: {'Enabled' if args.tune else 'Disabled'}")
    print(f"  CSV Path: {args.csv_path if args.csv_path else 'Auto-download from Kaggle'}")
    print("="*80)
    
    # Initialize trainer
    trainer = MLflowTrainer(
        experiment_name=args.experiment_name,
        random_state=args.random_state
    )
    
    # Run complete pipeline
    try:
        results = trainer.run_complete_pipeline(
            csv_path=args.csv_path,
            tune_models=args.tune
        )
        
        print_section_header("EXECUTION SUMMARY", width=80)
        print(f"✅ Pipeline completed successfully!")
        print(f"\n📊 Results:")
        print(f"  - Baseline Models Trained: {len(results['baseline_results'])}")
        print(f"  - Tuned Models: {len(results['tuned_results'])}")
        print(f"  - Ensemble Model: Created")
        
        print(f"\n🏆 Best Model:")
        best_model = results['comparison_df'].iloc[0]
        print(f"  - Model: {best_model['Model']}")
        print(f"  - Accuracy: {best_model['Accuracy']:.4f}")
        print(f"  - ROC AUC: {best_model['ROC AUC']:.4f}")
        
        print(f"\n📈 Ensemble Performance:")
        print(f"  - Accuracy: {results['ensemble_metrics']['accuracy']:.4f}")
        print(f"  - Precision: {results['ensemble_metrics']['precision']:.4f}")
        print(f"  - Recall: {results['ensemble_metrics']['recall']:.4f}")
        print(f"  - F1-Score: {results['ensemble_metrics']['f1_score']:.4f}")
        print(f"  - ROC AUC: {results['ensemble_metrics']['roc_auc']:.4f}")
        
        print(f"\n🔍 View Results:")
        print(f"  - MLflow UI: Run 'mlflow ui --port 5000' and open http://localhost:5000")
        print(f"  - Artifacts: Check 'artifacts/' directory")
        print(f"  - Summary Report: model_summary_report.txt")
        print(f"  - Training Log: training.log")
        
        print("\n" + "="*80)
        print("🎉 All tasks completed successfully!")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print_section_header("ERROR", width=80)
        print(f"❌ Pipeline execution failed!")
        print(f"Error: {str(e)}")
        print(f"\nPlease check the training.log file for details.")
        print("="*80)
        
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pima Indians Diabetes Classification - MLflow Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python main.py
  
  # Run without hyperparameter tuning (faster)
  python main.py --no-tune
  
  # Use custom experiment name
  python main.py --experiment-name "My_Diabetes_Experiment"
  
  # Use custom dataset
  python main.py --csv-path /path/to/diabetes.csv
  
  # Change random state for reproducibility
  python main.py --random-state 123
        """
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='Pima_Diabetes_Classification',
        help='MLflow experiment name (default: Pima_Diabetes_Classification)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--csv-path',
        type=str,
        default=None,
        help='Path to diabetes.csv file (default: auto-download from Kaggle)'
    )
    
    parser.add_argument(
        '--no-tune',
        dest='tune',
        action='store_false',
        help='Disable hyperparameter tuning (faster execution)'
    )
    
    parser.set_defaults(tune=True)
    
    args = parser.parse_args()
    
    # Execute main function
    exit_code = main(args)
    sys.exit(exit_code)
