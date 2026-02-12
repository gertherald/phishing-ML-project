"""
Model Trainer Module - Updated with Model-Specific Feature Selection
Handles multicollinearity appropriately for different model types.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost as xgb
import joblib
import pickle
from pathlib import Path
import logging
import time


class ModelTrainer:
    """Handles model training with hyperparameter tuning."""
    
    def __init__(self, config, preprocessor):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration object
            preprocessor: Fitted DataPreprocessor instance
        """
        self.config = config
        self.preprocessor = preprocessor
        self.logger = logging.getLogger(__name__)
        
        self.models = {}
        self.training_results = {}
        self.best_params = {}
    
    def _get_model_instance(self, model_name):
        """Get a model instance based on name."""
        model_config = self.config.MODELS.get(model_name)
        if not model_config or not model_config.get('enabled'):
            raise ValueError(f"Model {model_name} not found or not enabled")
        
        params = model_config['params'].copy()
        
        if model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_name == 'xgboost':
            return xgb.XGBClassifier(**params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'svm':
            return SVC(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_single_model(self, X_train, y_train, model_name, tune_hyperparameters=True):
        """
        Train a single model with optional hyperparameter tuning.
        
        CRITICAL: Applies model-specific feature selection before training.
        - Tree models: Use all features (including DomainAge_Squared)
        - Linear models: Exclude DomainAge_Squared (multicollinearity)
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of model to train
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Tuple of (trained_model, results_dict)
        """
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Training {model_name}")
        self.logger.info(f"{'='*80}")
        
        # CRITICAL: Apply model-specific feature selection
        X_train_filtered = self.preprocessor.get_features_for_model(X_train, model_name)
        self.logger.info(f"Feature set shape: {X_train_filtered.shape}")

        if not hasattr(self, 'feature_names_'):
            self.feature_names_ = {}
        self.feature_names_[model_name] = X_train_filtered.columns.tolist()
        self.logger.info(f"Stored {len(self.feature_names_[model_name])} feature names for {model_name}")

        
        start_time = time.time()
        
        # Get base model
        model = self._get_model_instance(model_name)
        
        results = {
            'model_name': model_name,
            'training_time': 0,
            'cv_score_mean': None,
            'cv_score_std': None,
            'best_params': None,
            'feature_count': X_train_filtered.shape[1]
        }
        
        # Hyperparameter tuning
        if tune_hyperparameters and model_name in self.config.PARAM_GRIDS:
            self.logger.info(f"Performing hyperparameter tuning ({self.config.TUNING_METHOD} search)")
            
            param_grid = self.config.PARAM_GRIDS[model_name]
            
            if self.config.TUNING_METHOD == 'grid':
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=self.config.CV_FOLDS,
                    scoring=self.config.TUNING_SCORING,
                    n_jobs=-1,
                    verbose=1
                )
            else:  # random search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=self.config.N_ITER_RANDOM_SEARCH,
                    cv=self.config.CV_FOLDS,
                    scoring=self.config.TUNING_SCORING,
                    n_jobs=-1,
                    verbose=1,
                    random_state=self.config.RANDOM_STATE
                )
            
            search.fit(X_train_filtered, y_train)
            
            model = search.best_estimator_
            results['best_params'] = search.best_params_
            results['cv_score_mean'] = search.best_score_
            results['cv_score_std'] = search.cv_results_['std_test_score'][search.best_index_]
            
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best CV {self.config.TUNING_SCORING}: {search.best_score_:.4f}")
            
        else:
            # Train with default parameters
            self.logger.info("Training with default parameters")
            
            # Cross-validation to estimate performance
            cv_scores = cross_val_score(
                model, 
                X_train_filtered, 
                y_train,
                cv=self.config.CV_FOLDS,
                scoring=self.config.TUNING_SCORING,
                n_jobs=-1
            )
            
            results['cv_score_mean'] = cv_scores.mean()
            results['cv_score_std'] = cv_scores.std()
            
            self.logger.info(f"CV {self.config.TUNING_SCORING}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Train on full training set
            model.fit(X_train_filtered, y_train)
        
        training_time = time.time() - start_time
        results['training_time'] = training_time
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"{'='*80}\n")
        
        # Store results
        self.models[model_name] = model
        self.training_results[model_name] = results
        if results['best_params']:
            self.best_params[model_name] = results['best_params']
        
        return model, results
    
    def train_all_models(self, X_train, y_train):
        """
        Train all enabled models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.logger.info("="*80)
        self.logger.info("TRAINING ALL ENABLED MODELS")
        self.logger.info("="*80)
        
        enabled_models = self.config.get_enabled_models()
        self.logger.info(f"Enabled models: {enabled_models}")
        
        for model_name in enabled_models:
            model_config = self.config.MODELS[model_name]
            tune = model_config.get('tune_hyperparameters', False)
            
            try:
                self.train_single_model(X_train, y_train, model_name, tune_hyperparameters=tune)
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        self.logger.info("="*80)
        self.logger.info("ALL MODELS TRAINED")
        self.logger.info("="*80)
    
    def predict(self, X, model_name):
        """
        Make predictions with a trained model.
        
        CRITICAL: Applies same model-specific feature selection as training.
        
        Args:
            X: Features to predict on
            model_name: Name of model to use
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Apply same feature filtering as training
        X_filtered = self.preprocessor.get_features_for_model(X, model_name)
        
        model = self.models[model_name]
        return model.predict(X_filtered)
    
    def predict_proba(self, X, model_name):
        """Get prediction probabilities."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        # Apply same feature filtering as training
        X_filtered = self.preprocessor.get_features_for_model(X, model_name)
        
        model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_filtered)
        else:
            raise ValueError(f"Model {model_name} does not support predict_proba")
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of model
            feature_names: Optional list of feature names (if None, will get from last training)
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get the actual feature names used for this model
            if feature_names is None:
                # Feature names should be stored during training
                # For now, try to get from the model's n_features_in_
                if hasattr(self, 'feature_names_') and model_name in self.feature_names_:
                    feature_names = self.feature_names_[model_name]
                else:
                    # Fallback: use generic names
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
                    self.logger.warning(f"Using generic feature names for {model_name}")
            
            # Ensure feature_names matches importances length
            if len(feature_names) != len(importances):
                self.logger.warning(
                    f"Feature names length ({len(feature_names)}) != "
                    f"importances length ({len(importances)}). Using generic names."
                )
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        elif hasattr(model, 'coef_'):
            # For linear models
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_[0])
            else:
                importances = np.abs(model.coef_)
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            self.logger.warning(f"Model {model_name} does not provide feature importances")
        return None
    
    def select_best_model(self, X_val, y_val, metric='f1'):
        """
        Select best model based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to use for selection
            
        Returns:
            Name of best model
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metric_functions = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, 
                self.predict_proba(X_val, best_model_name)[:, 1])
        }
        
        best_score = -np.inf
        best_model_name = None
        
        for model_name in self.models.keys():
            y_pred = self.predict(X_val, model_name)
            
            if metric == 'roc_auc':
                if hasattr(self.models[model_name], 'predict_proba'):
                    y_pred_proba = self.predict_proba(X_val, model_name)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                else:
                    continue
            else:
                score = metric_functions[metric](y_val, y_pred)
            
            self.logger.info(f"{model_name} validation {metric}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.logger.info(f"\nBest model: {best_model_name} ({metric}={best_score:.4f})")
        return best_model_name
    
    def save_model(self, model_name, filepath):
        """Save a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
        
        model = self.models[model_name]
        
        if self.config.MODEL_FILE_FORMAT == 'joblib':
            joblib.dump(model, filepath)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        self.logger.info(f"Saved {model_name} to {filepath}")
    
    def load_model(self, model_name, filepath):
        """Load a trained model."""
        if self.config.MODEL_FILE_FORMAT == 'joblib':
            model = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        
        self.models[model_name] = model
        self.logger.info(f"Loaded {model_name} from {filepath}")
        return model
    
    def get_training_summary(self):
        """Get summary of training results."""
        if not self.training_results:
            return None
        
        summary = pd.DataFrame(self.training_results).T
        summary = summary.sort_values('cv_score_mean', ascending=False)
        return summary
