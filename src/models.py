"""
Model Definitions for Pima Indians Diabetes Classification
Includes KNN, LightGBM, Logistic Regression, Random Forest, XGBoost, SVM, and Neural Network
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint, uniform


class ModelFactory:
    """Factory class to create and configure ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
    
    def get_all_models(self):
        """Get all models with default configurations"""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000,
                solver='lbfgs'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='euclidean'
            ),
            'SVM': SVC(
                random_state=self.random_state, 
                probability=True,
                kernel='rbf',
                C=1.0,
                gamma='scale'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=2
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state, 
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state, 
                eval_metric='logloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                n_jobs=-1
            ),
            'LightGBM': LGBMClassifier(
                random_state=self.random_state, 
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                n_jobs=-1
            ),
            'Neural Network': MLPClassifier(
                random_state=self.random_state,
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        }
        return models
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        models = self.get_all_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
        return models[model_name]
    
    def get_param_grids(self):
        """Get hyperparameter grids for GridSearchCV/RandomizedSearchCV"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'KNN': {
                'n_neighbors': range(1, 31, 2),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'Decision Tree': {
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300, 500, 1000],
                'num_leaves': randint(6, 50),
                'min_child_samples': randint(100, 500),
                'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1],
                'subsample': uniform(loc=0.2, scale=0.8),
                'max_depth': [-1, 1, 2, 3, 5, 7],
                'colsample_bytree': uniform(loc=0.4, scale=0.6),
                'reg_alpha': [0, 0.1, 1, 2, 5, 10],
                'reg_lambda': [0, 0.1, 1, 5, 10, 20]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'learning_rate_init': [0.001, 0.01, 0.1]
            }
        }
        return param_grids
    
    def get_param_distributions(self):
        """Get parameter distributions for RandomizedSearchCV"""
        param_distributions = {
            'LightGBM': {
                'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],
                'n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],
                'num_leaves': randint(6, 50),
                'min_child_samples': randint(100, 500),
                'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                'subsample': uniform(loc=0.2, scale=0.8),
                'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7],
                'colsample_bytree': uniform(loc=0.4, scale=0.6),
                'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
            },
            'Random Forest': {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None]
            },
            'XGBoost': {
                'n_estimators': randint(50, 300),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_child_weight': randint(1, 5),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4)
            }
        }
        return param_distributions


def get_priority_models():
    """Get priority models (KNN and LightGBM as specified)"""
    factory = ModelFactory()
    models = factory.get_all_models()
    return {
        'KNN': models['KNN'],
        'LightGBM': models['LightGBM']
    }


def get_all_classification_models(random_state=42):
    """Convenience function to get all models"""
    factory = ModelFactory(random_state=random_state)
    return factory.get_all_models()


if __name__ == "__main__":
    # Test model factory
    factory = ModelFactory()
    models = factory.get_all_models()
    
    print("Available Models:")
    print("="*50)
    for name, model in models.items():
        print(f"- {name}: {type(model).__name__}")
    
    print(f"\nTotal models: {len(models)}")
    
    print("\nParameter grids available for:")
    param_grids = factory.get_param_grids()
    for name in param_grids.keys():
        print(f"- {name}")
