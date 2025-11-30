"""
Training Module with MLflow Integration for Pima Indians Diabetes Classification
Complete training pipeline with MLflow tracking, autologging, and hyperparameter tuning
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
import optuna
from optuna.integration.mlflow import MLflowCallback

warnings.filterwarnings('ignore')

# Import project modules
from preprocess import PimaPreprocessor, get_preprocessed_data
from models import ModelFactory, get_all_classification_models, get_priority_models
from evaluation import ModelEvaluator
from utils import timer, save_model, create_summary_report, ExperimentLogger, print_section_header


class MLflowTrainer:
    """Main trainer class with complete MLflow integration"""
    
    def __init__(self, experiment_name="Pima_Diabetes_Classification", random_state=42):
        self.experiment_name = experiment_name
        self.random_state = random_state
        self.model_factory = ModelFactory(random_state=random_state)
        self.evaluator = ModelEvaluator(artifact_dir='artifacts')
        self.logger = ExperimentLogger(log_file='training.log')
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
        
        # Enable autologging for sklearn
        mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
        
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment: {experiment_name}")
    
    def train_single_model(self, model_name, model, X_train, y_train, X_test, y_test, feature_names):
        """Train a single model with MLflow tracking"""
        self.logger.log(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            # Log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("random_state", self.random_state)
            
            # Train model
            with timer(f"{model_name} training"):
                model.fit(X_train, y_train)
            
            # Evaluate model
            metrics, artifacts, report = self.evaluator.evaluate_model(
                model, X_test, y_test, model_name, feature_names, log_to_mlflow=True
            )
            
            # Log model
            mlflow.sklearn.log_model(model, f"model_{model_name}")
            
            # Add tags
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("model_family", model_name)
            
            self.logger.log_metrics(model_name, metrics)
            
        return model, metrics, artifacts
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names):
        """Train all models and collect results"""
        print_section_header("TRAINING ALL MODELS")
        
        models = self.model_factory.get_all_models()
        results = {}
        trained_models = {}
        
        for model_name, model in models.items():
            try:
                trained_model, metrics, artifacts = self.train_single_model(
                    model_name, model, X_train, y_train, X_test, y_test, feature_names
                )
                results[model_name] = metrics
                trained_models[model_name] = trained_model
            except Exception as e:
                self.logger.log(f"Error training {model_name}: {str(e)}", level='ERROR')
                continue
        
        # Compare models
        comparison_df, csv_path, viz_path = self.evaluator.compare_models(results)
        
        # Log comparison as artifacts
        with mlflow.start_run(run_name="Model_Comparison"):
            mlflow.log_artifact(csv_path)
            mlflow.log_artifact(viz_path)
            
            # Log best model metrics
            best_model = comparison_df.iloc[0]
            mlflow.log_metric("best_accuracy", best_model['Accuracy'])
            mlflow.log_param("best_model", best_model['Model'])
        
        print_section_header("MODEL TRAINING COMPLETED")
        print(comparison_df.to_string(index=False))
        
        return trained_models, results, comparison_df
    
    def hyperparameter_tuning_grid(self, model_name, X_train, y_train, X_test, y_test, feature_names):
        """Hyperparameter tuning using GridSearchCV with MLflow tracking"""
        self.logger.log(f"GridSearchCV tuning for {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_GridSearchCV"):
            model = self.model_factory.get_model(model_name)
            param_grid = self.model_factory.get_param_grids()[model_name]
            
            # Limit grid for faster execution
            if model_name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            elif model_name == 'XGBoost':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            
            mlflow.log_param("tuning_method", "GridSearchCV")
            mlflow.log_param("param_grid", str(param_grid))
            
            with timer(f"GridSearchCV for {model_name}"):
                grid_search = GridSearchCV(
                    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("cv_best_score", grid_search.best_score_)
            
            # Evaluate on test set
            best_model = grid_search.best_estimator_
            metrics, artifacts, _ = self.evaluator.evaluate_model(
                best_model, X_test, y_test, f"{model_name}_tuned", feature_names
            )
            
            mlflow.sklearn.log_model(best_model, f"model_{model_name}_tuned")
            mlflow.set_tag("stage", "tuned")
            
            self.logger.log(f"Best CV score: {grid_search.best_score_:.4f}")
            self.logger.log(f"Test accuracy: {metrics['accuracy']:.4f}")
            
        return best_model, metrics
    
    def hyperparameter_tuning_random(self, model_name, X_train, y_train, X_test, y_test, feature_names, n_iter=50):
        """Hyperparameter tuning using RandomizedSearchCV with MLflow tracking"""
        self.logger.log(f"RandomizedSearchCV tuning for {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_RandomSearchCV"):
            model = self.model_factory.get_model(model_name)
            param_distributions = self.model_factory.get_param_distributions()
            
            if model_name not in param_distributions:
                self.logger.log(f"No param distributions for {model_name}", level='WARNING')
                return None, None
            
            mlflow.log_param("tuning_method", "RandomizedSearchCV")
            mlflow.log_param("n_iter", n_iter)
            
            with timer(f"RandomizedSearchCV for {model_name}"):
                random_search = RandomizedSearchCV(
                    model, param_distributions[model_name], 
                    n_iter=n_iter, cv=5, scoring='accuracy', 
                    random_state=self.random_state, n_jobs=-1, verbose=0
                )
                random_search.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric("cv_best_score", random_search.best_score_)
            
            # Evaluate on test set
            best_model = random_search.best_estimator_
            metrics, artifacts, _ = self.evaluator.evaluate_model(
                best_model, X_test, y_test, f"{model_name}_random_tuned", feature_names
            )
            
            mlflow.sklearn.log_model(best_model, f"model_{model_name}_random_tuned")
            mlflow.set_tag("stage", "random_tuned")
            
            self.logger.log(f"Best CV score: {random_search.best_score_:.4f}")
            self.logger.log(f"Test accuracy: {metrics['accuracy']:.4f}")
            
        return best_model, metrics
    
    def hyperparameter_tuning_optuna(self, model_name, X_train, y_train, X_test, y_test, feature_names, n_trials=50):
        """Hyperparameter tuning using Optuna with MLflow tracking"""
        self.logger.log(f"Optuna tuning for {model_name}...")
        
        mlflc = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="accuracy")
        
        with mlflow.start_run(run_name=f"{model_name}_Optuna"):
            mlflow.log_param("tuning_method", "Optuna")
            mlflow.log_param("n_trials", n_trials)
            
            def objective(trial):
                if model_name == 'Random Forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'random_state': self.random_state,
                        'n_jobs': -1
                    }
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**params)
                    
                elif model_name == 'XGBoost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'random_state': self.random_state,
                        'eval_metric': 'logloss',
                        'n_jobs': -1
                    }
                    from xgboost import XGBClassifier
                    model = XGBClassifier(**params)
                    
                elif model_name == 'LightGBM':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                        'random_state': self.random_state,
                        'verbose': -1,
                        'n_jobs': -1
                    }
                    from lightgbm import LGBMClassifier
                    model = LGBMClassifier(**params)
                else:
                    return 0
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                return cv_scores.mean()
            
            with timer(f"Optuna optimization for {model_name}"):
                study = optuna.create_study(
                    direction='maximize', 
                    sampler=optuna.samplers.TPESampler(seed=self.random_state)
                )
                study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])
            
            # Train final model with best params
            best_params = study.best_params
            best_params['random_state'] = self.random_state
            
            if model_name == 'Random Forest':
                from sklearn.ensemble import RandomForestClassifier
                best_model = RandomForestClassifier(**best_params, n_jobs=-1)
            elif model_name == 'XGBoost':
                from xgboost import XGBClassifier
                best_params['eval_metric'] = 'logloss'
                best_model = XGBClassifier(**best_params, n_jobs=-1)
            elif model_name == 'LightGBM':
                from lightgbm import LGBMClassifier
                best_params['verbose'] = -1
                best_model = LGBMClassifier(**best_params, n_jobs=-1)
            
            best_model.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(best_params)
            mlflow.log_metric("optuna_best_value", study.best_value)
            
            # Evaluate on test set
            metrics, artifacts, _ = self.evaluator.evaluate_model(
                best_model, X_test, y_test, f"{model_name}_optuna_tuned", feature_names
            )
            
            mlflow.sklearn.log_model(best_model, f"model_{model_name}_optuna_tuned")
            mlflow.set_tag("stage", "optuna_tuned")
            
            self.logger.log(f"Best Optuna value: {study.best_value:.4f}")
            self.logger.log(f"Test accuracy: {metrics['accuracy']:.4f}")
            
        return best_model, metrics
    
    def create_ensemble(self, models_dict, X_train, y_train, X_test, y_test, feature_names):
        """Create and evaluate ensemble model"""
        self.logger.log("Creating ensemble model...")
        
        with mlflow.start_run(run_name="Ensemble_VotingClassifier"):
            # Create ensemble with top models
            estimators = [(name, model) for name, model in models_dict.items()]
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                n_jobs=-1
            )
            
            mlflow.log_param("ensemble_type", "VotingClassifier")
            mlflow.log_param("voting", "soft")
            mlflow.log_param("n_models", len(estimators))
            mlflow.log_param("models", str([name for name, _ in estimators]))
            
            with timer("Training ensemble"):
                ensemble.fit(X_train, y_train)
            
            # Evaluate ensemble
            metrics, artifacts, _ = self.evaluator.evaluate_model(
                ensemble, X_test, y_test, "Ensemble", feature_names
            )
            
            mlflow.sklearn.log_model(ensemble, "ensemble_model")
            mlflow.set_tag("stage", "ensemble")
            
            self.logger.log_metrics("Ensemble", metrics)
            
        return ensemble, metrics
    
    def run_complete_pipeline(self, csv_path=None, tune_models=True):
        """Run the complete ML pipeline with MLflow tracking"""
        print_section_header("STARTING COMPLETE MLFLOW PIPELINE")
        
        # 1. Data preprocessing
        print_section_header("STEP 1: DATA PREPROCESSING")
        preprocessor = PimaPreprocessor(random_state=self.random_state)
        X_train, X_test, y_train, y_test, full_data = preprocessor.preprocess_pipeline(csv_path)
        feature_names = X_train.columns.tolist()
        
        with mlflow.start_run(run_name="Data_Preprocessing"):
            mlflow.log_param("n_train_samples", X_train.shape[0])
            mlflow.log_param("n_test_samples", X_test.shape[0])
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", self.random_state)
        
        # 2. Train all models
        print_section_header("STEP 2: TRAINING ALL BASELINE MODELS")
        trained_models, baseline_results, comparison_df = self.train_all_models(
            X_train, y_train, X_test, y_test, feature_names
        )
        
        # 3. Hyperparameter tuning for priority models
        tuned_models = {}
        tuned_results = {}
        
        if tune_models:
            print_section_header("STEP 3: HYPERPARAMETER TUNING")
            
            # Get top 3 models for tuning
            top_3_models = comparison_df.head(3)['Model'].tolist()
            
            for model_name in top_3_models:
                if model_name in ['KNN']:
                    model, metrics = self.hyperparameter_tuning_grid(
                        model_name, X_train, y_train, X_test, y_test, feature_names
                    )
                elif model_name in ['LightGBM']:
                    model, metrics = self.hyperparameter_tuning_random(
                        model_name, X_train, y_train, X_test, y_test, feature_names, n_iter=50
                    )
                elif model_name in ['Random Forest', 'XGBoost']:
                    model, metrics = self.hyperparameter_tuning_optuna(
                        model_name, X_train, y_train, X_test, y_test, feature_names, n_trials=30
                    )
                else:
                    continue
                
                if model and metrics:
                    tuned_models[model_name] = model
                    tuned_results[model_name] = metrics
        
        # 4. Create ensemble with best models
        print_section_header("STEP 4: CREATING ENSEMBLE MODEL")
        if tuned_models:
            ensemble_models = tuned_models
        else:
            # Use top 2 baseline models
            top_2 = comparison_df.head(2)['Model'].tolist()
            ensemble_models = {name: trained_models[name] for name in top_2}
        
        ensemble_model, ensemble_metrics = self.create_ensemble(
            ensemble_models, X_train, y_train, X_test, y_test, feature_names
        )
        
        # 5. Final summary
        print_section_header("STEP 5: FINAL RESULTS SUMMARY")
        all_results = {**baseline_results, **tuned_results, "Ensemble": ensemble_metrics}
        
        summary_path = create_summary_report(all_results, 'model_summary_report.txt')
        
        with mlflow.start_run(run_name="Final_Summary"):
            mlflow.log_artifact(summary_path)
            
            # Log best overall metrics
            best_model = max(all_results.items(), key=lambda x: x[1].get('accuracy', 0))
            mlflow.log_param("best_overall_model", best_model[0])
            mlflow.log_metric("best_overall_accuracy", best_model[1]['accuracy'])
        
        self.logger.log_elapsed_time()
        print_section_header("PIPELINE COMPLETED SUCCESSFULLY")
        
        print(f"\n{mlflow.get_artifact_uri()}")
        print(f"View results: mlflow ui --port 5000")
        
        return {
            'trained_models': trained_models,
            'tuned_models': tuned_models,
            'ensemble_model': ensemble_model,
            'baseline_results': baseline_results,
            'tuned_results': tuned_results,
            'ensemble_metrics': ensemble_metrics,
            'comparison_df': comparison_df
        }


if __name__ == "__main__":
    # Run complete pipeline
    trainer = MLflowTrainer(
        experiment_name="Pima_Diabetes_Classification",
        random_state=42
    )
    
    results = trainer.run_complete_pipeline(csv_path=None, tune_models=True)
    
    print("\nAll models trained and logged to MLflow!")
    print("Run 'mlflow ui' to view the results in the MLflow UI")
