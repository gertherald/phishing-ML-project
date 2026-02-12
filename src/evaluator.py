"""
Model Evaluator Module

WHAT IT DOES:
- Evaluates trained models on test data
- Calculates performance metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generates confusion matrices and ROC curves
- Visualizes feature importance
- Compares all models and selects the best one
- Saves evaluation results

WHY WE NEED IT:
- Comprehensive model assessment
- Visual performance analysis
- Model comparison for selection
- Production deployment decision support
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import logging
from pathlib import Path
import json


class ModelEvaluator:
    """Handles model evaluation and performance analysis."""
    
    def __init__(self, config):
        """
        Initialize the evaluator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.predictions = {}
        
    def evaluate_single_model(self, y_true, y_pred, y_pred_proba, model_name):
        """
        Evaluate a single model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            model_name: Name of the model
            
        Returns:
            Dict: Performance metrics
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        # ROC-AUC (if probabilities available)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = cm[0, 0]
        metrics['fp'] = cm[0, 1]
        metrics['fn'] = cm[1, 0]
        metrics['tp'] = cm[1, 1]
        
        # Store results
        self.results[model_name] = metrics
        self.predictions[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Log metrics
        self.logger.info(f"{model_name} Results:")
        self.logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            self.logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test, trainer):
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features (BEFORE feature selection)
            y_test: Test labels
            trainer: Trained ModelTrainer instance
        """
        self.logger.info("="*80)
        self.logger.info("EVALUATING ALL MODELS")
        self.logger.info("="*80)
        
        for model_name in trainer.models.keys():
            try:
                # Make predictions (trainer handles feature selection internally)
                y_pred = trainer.predict(X_test, model_name)
                
                # Get probabilities if available
                try:
                    y_pred_proba = trainer.predict_proba(X_test, model_name)[:, 1]
                except:
                    y_pred_proba = None
                
                # Evaluate
                self.evaluate_single_model(y_test, y_pred, y_pred_proba, model_name)
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        self.logger.info("="*80)
    
    def compare_models(self):
        """
        Compare all evaluated models.
        
        Returns:
            pd.DataFrame: Comparison table sorted by F1 score
        """
        if not self.results:
            self.logger.warning("No results to compare")
            return None
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'ROC-AUC': metrics.get('roc_auc', np.nan)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1 Score', ascending=False)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("="*80)
        self.logger.info("\n" + df.to_string(index=False))
        
        return df
    
    def select_best_model(self, metric='f1'):
        """
        Select best model based on specified metric.
        
        Args:
            metric: Metric to use for selection (default: 'f1')
            
        Returns:
            str: Name of best model
        """
        if not self.results:
            raise ValueError("No models evaluated yet")
        
        best_score = -np.inf
        best_model = None
        
        for model_name, metrics in self.results.items():
            score = metrics.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model = model_name
        
        self.logger.info(f"\nBest model: {best_model} ({metric}={best_score:.4f})")
        return best_model
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """Plot confusion matrix for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not evaluated")
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Phishing', 'Legitimate'],
                    yticklabels=['Phishing', 'Legitimate'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, model_name, y_test, save_path=None):
        """Plot ROC curve for a model."""
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not evaluated")
        
        y_pred_proba = self.predictions[model_name]['y_pred_proba']
        if y_pred_proba is None:
            self.logger.warning(f"No probabilities available for {model_name}")
            return
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = self.results[model_name]['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ROC curve to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, model_name, trainer, top_n=10, save_path=None):
        """Plot feature importance for tree-based models."""
        importance_df = trainer.get_feature_importance(model_name)
        
        if importance_df is None:
            self.logger.warning(f"No feature importance available for {model_name}")
            return
        
        # Plot top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved feature importance to {save_path}")
        
        plt.close()
    
    def generate_classification_report(self, model_name, y_test):
        """Generate detailed classification report."""
        if model_name not in self.predictions:
            raise ValueError(f"Model {model_name} not evaluated")
        
        y_pred = self.predictions[model_name]['y_pred']
        report = classification_report(y_test, y_pred, 
                                       target_names=['Phishing', 'Legitimate'])
        
        self.logger.info(f"\nClassification Report - {model_name}:")
        self.logger.info("\n" + report)
        
        return report
    
    def save_results(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table (this already worked)
        comparison_df = self.compare_models()
        if comparison_df is not None:
            comparison_path = output_dir / 'model_comparison.csv'
            comparison_df.to_csv(comparison_path, index=False)
            self.logger.info(f"Saved model comparison to {comparison_path}")
        
        # Skip JSON saving - CSV is enough
        self.logger.info("Results saved successfully")
    
    def generate_report(self, y_test, trainer, output_dir):
        """
        Generate comprehensive evaluation report with visualizations.
        
        Args:
            y_test: Test labels
            trainer: Trained ModelTrainer instance
            output_dir: Directory to save results
        """
        self.logger.info("Generating comprehensive evaluation report")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        self.save_results(output_dir)
        
        # Generate visualizations for each model
        for model_name in self.results.keys():
            # Confusion matrix
            cm_path = output_dir / f'confusion_matrix_{model_name}.png'
            self.plot_confusion_matrix(model_name, save_path=cm_path)
            
            # ROC curve
            roc_path = output_dir / f'roc_curve_{model_name}.png'
            self.plot_roc_curve(model_name, y_test, save_path=roc_path)
            
            # Feature importance (for tree models)
            if model_name in self.config.TREE_BASED_MODELS:
                fi_path = output_dir / f'feature_importance_{model_name}.png'
                self.plot_feature_importance(model_name, trainer, save_path=fi_path)
            
            # Classification report
            report = self.generate_classification_report(model_name, y_test)
            report_path = output_dir / f'classification_report_{model_name}.txt'
            with open(report_path, 'w') as f:
                f.write(report)
        
        self.logger.info(f"Evaluation report saved to {output_dir}")
