"""
ML Pipeline Orchestrator

WHAT IT DOES:
- Coordinates the entire ML workflow from start to finish
- Calls all modules in the correct order
- Handles errors and logging
- Saves intermediate and final results
- Generates comprehensive reports

WHY WE NEED IT:
- Single entry point for the complete pipeline
- Ensures reproducible workflow
- Error handling and recovery
- Progress tracking and logging
"""

import logging
import time
from pathlib import Path
import joblib
import numpy as np

from config import Config
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from evaluator import ModelEvaluator


class PhishingDetectionPipeline:
    """Main pipeline orchestrator for phishing detection."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.config = Config
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Components (initialized during run)
        self.data_loader = None
        self.preprocessor = None
        self.trainer = None
        self.evaluator = None
        
        # Results
        self.results = {}
        
    def setup_logging(self):
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=getattr(logging, self.config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.OUTPUT_DIR / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
    
    def run(self):
        """
        Execute the complete ML pipeline.
        
        Steps:
        1. Load data from database
        2. Split into train/val/test
        3. Preprocess data (fit on train, transform on val/test)
        4. Select top 10 features (model-specific)
        5. Train models
        6. Evaluate models
        7. Select best model
        8. Save everything
        9. Generate report
        
        Returns:
            Dict: Pipeline results including best model and metrics
        """
        start_time = time.time()
        
        self.logger.info("="*80)
        self.logger.info("PHISHING DETECTION ML PIPELINE - START")
        self.logger.info("="*80)
        
        try:
            # Create output directories
            self.config.create_directories()
            
            # Step 1: Load Data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 1: LOADING DATA")
            self.logger.info("="*80)
            self.data_loader = DataLoader(self.config)
            df = self.data_loader.load_data()
            
            # Get data summary
            summary = self.data_loader.get_data_summary(df)
            self.logger.info(f"Dataset summary: {summary}")
            
            # Step 2: Split Data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 2: SPLITTING DATA")
            self.logger.info("="*80)
            train_df, val_df, test_df = self.data_loader.split_data(df)
            
            # Step 3: Preprocess Data
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 3: PREPROCESSING DATA")
            self.logger.info("="*80)
            
            self.preprocessor = DataPreprocessor(self.config)
            
            # Fit and transform training data
            self.logger.info("Processing training data (fit + transform)...")
            train_processed = self.preprocessor.fit_transform(train_df)
            X_train, y_train = self.preprocessor.split_features_target(train_processed)
            
            # Transform validation data
            self.logger.info("Processing validation data (transform only)...")
            val_processed = self.preprocessor.transform(val_df)
            X_val, y_val = self.preprocessor.split_features_target(val_processed)
            
            # Transform test data
            self.logger.info("Processing test data (transform only)...")
            test_processed = self.preprocessor.transform(test_df)
            X_test, y_test = self.preprocessor.split_features_target(test_processed)
            
            self.logger.info(f"Processed shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            
            # Step 4: Train Models
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 4: TRAINING MODELS")
            self.logger.info("="*80)
            self.logger.info("NOTE: Each model will use only its TOP 10 features")
            self.logger.info("  - Tree models: Include DomainAge_Squared")
            self.logger.info("  - Linear models: Exclude DomainAge_Squared")
            self.logger.info("="*80)
            
            self.trainer = ModelTrainer(self.config, self.preprocessor)
            self.trainer.train_all_models(X_train, y_train)
            
            # Get training summary
            training_summary = self.trainer.get_training_summary()
            if training_summary is not None:
                self.logger.info("\nTraining Summary:")
                self.logger.info("\n" + training_summary.to_string())
            
            # Step 5: Evaluate Models
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 5: EVALUATING MODELS")
            self.logger.info("="*80)
            
            self.evaluator = ModelEvaluator(self.config)
            self.evaluator.evaluate_all_models(X_test, y_test, self.trainer)
            
            # Compare models
            comparison_df = self.evaluator.compare_models()
            
            # Step 6: Select Best Model
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 6: SELECTING BEST MODEL")
            self.logger.info("="*80)
            
            best_model_name = self.evaluator.select_best_model(metric=self.config.PRIMARY_METRIC)
            best_metrics = self.evaluator.results[best_model_name]
            
            self.logger.info(f"\nBest Model: {best_model_name}")
            self.logger.info(f"  Accuracy:  {best_metrics['accuracy']:.4f}")
            self.logger.info(f"  Precision: {best_metrics['precision']:.4f}")
            self.logger.info(f"  Recall:    {best_metrics['recall']:.4f}")
            self.logger.info(f"  F1 Score:  {best_metrics['f1']:.4f}")
            if 'roc_auc' in best_metrics:
                self.logger.info(f"  ROC-AUC:   {best_metrics['roc_auc']:.4f}")
            
            # Step 7: Save Models and Preprocessor
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 7: SAVING MODELS")
            self.logger.info("="*80)
            
            if self.config.SAVE_MODELS:
                for model_name in self.trainer.models.keys():
                    model_path = self.config.MODEL_DIR / f'{model_name}.joblib'
                    self.trainer.save_model(model_name, model_path)
            
            if self.config.SAVE_PREPROCESSORS:
                preprocessor_path = self.config.MODEL_DIR / 'preprocessor.joblib'
                self.preprocessor.save(preprocessor_path)
            
            # Step 8: Generate Comprehensive Report
            self.logger.info("\n" + "="*80)
            self.logger.info("STEP 8: GENERATING EVALUATION REPORT")
            self.logger.info("="*80)
            
            self.evaluator.generate_report(y_test, self.trainer, self.config.RESULTS_DIR)
            
            # Step 9: Save Final Results
            self.results = {
                'best_model': best_model_name,
                'best_metrics': best_metrics,
                'all_results': self.evaluator.results,
                'comparison': comparison_df.to_dict() if comparison_df is not None else None,
                'data_summary': summary,
                'training_time_seconds': time.time() - start_time
            }
            
            self.logger.info("Results saved to model_comparison.csv")
            
            # Final summary
            elapsed_time = time.time() - start_time
            self.logger.info("\n" + "="*80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            self.logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
            self.logger.info(f"Best model: {best_model_name}")
            self.logger.info(f"F1 Score: {best_metrics['f1']:.4f}")
            self.logger.info(f"Results saved to: {self.config.RESULTS_DIR}")
            self.logger.info("="*80)
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.exception("Full traceback:")
            raise
    
    def save_pipeline(self, path):
        """Save entire pipeline (preprocessor + all models)."""
        pipeline_data = {
            'config': self.config,
            'preprocessor': self.preprocessor,
            'models': self.trainer.models,
            'results': self.results
        }
        joblib.dump(pipeline_data, path)
        self.logger.info(f"Pipeline saved to {path}")
    
    def load_pipeline(self, path):
        """Load saved pipeline."""
        pipeline_data = joblib.load(path)
        self.config = pipeline_data['config']
        self.preprocessor = pipeline_data['preprocessor']
        self.trainer = ModelTrainer(self.config, self.preprocessor)
        self.trainer.models = pipeline_data['models']
        self.results = pipeline_data['results']
        self.logger.info(f"Pipeline loaded from {path}")
