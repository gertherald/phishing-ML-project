"""
Main Execution Script

WHAT IT DOES:
- Entry point for the phishing detection ML pipeline
- Sets up environment and logging
- Executes the complete pipeline
- Handles errors and reports results
- Provides clean command-line interface

WHY WE NEED IT:
- Simple execution: python run.py
- User-friendly output
- Error handling
- Exit codes for automation

USAGE:
    python run.py
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import PhishingDetectionPipeline
from config import Config


def print_header():
    """Print pipeline header."""
    print("\n" + "="*80)
    print(" "*20 + "PHISHING DETECTION ML PIPELINE")
    print("="*80)
    print("\nObjective: Train and evaluate ML models for phishing website detection")
    print("Strategy: Top 10 features only, model-specific feature selection")
    print("="*80 + "\n")


def setup_logging():
    """Configure logging."""
    Config.create_directories()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.OUTPUT_DIR / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )


def print_results(results):
    """Print final results in a user-friendly format."""
    print("\n" + "="*80)
    print(" "*25 + "PIPELINE RESULTS")
    print("="*80)
    
    print(f"\n✅ Best Model: {results['best_model'].upper()}")
    print("\nPerformance Metrics:")
    print("-" * 40)
    
    metrics = results['best_metrics']
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 40)
    print(f"  True Negatives (TN):  {metrics['tn']:,}")
    print(f"  False Positives (FP): {metrics['fp']:,}")
    print(f"  False Negatives (FN): {metrics['fn']:,}")
    print(f"  True Positives (TP):  {metrics['tp']:,}")
    
    print("\nData Summary:")
    print("-" * 40)
    summary = results['data_summary']
    print(f"  Total Samples: {summary['total_samples']:,}")
    print(f"  Phishing:      {summary['n_phishing']:,} ({summary['phishing_percentage']:.1f}%)")
    print(f"  Legitimate:    {summary['n_legitimate']:,} ({summary['legitimate_percentage']:.1f}%)")
    
    print("\nExecution:")
    print("-" * 40)
    elapsed = results['training_time_seconds']
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"  Total Time: {minutes}m {seconds:.1f}s")
    
    print("\nOutput Locations:")
    print("-" * 40)
    print(f"  Models:  {Config.MODEL_DIR}")
    print(f"  Results: {Config.RESULTS_DIR}")
    
    print("\n" + "="*80)
    print(" "*20 + "✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


def main():
    """Main execution function."""
    try:
        # Print header
        print_header()
        
        # Setup
        setup_logging()
        Config.create_directories()
        
        # Run pipeline
        print("Starting pipeline execution...\n")
        pipeline = PhishingDetectionPipeline()
        results = pipeline.run()
        
        # Print results
        print_results(results)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found")
        print(f"   {str(e)}")
        print(f"\nPlease ensure the database file exists at: {Config.DB_PATH}")
        return 1
        
    except ValueError as e:
        print(f"\n❌ ERROR: Invalid data")
        print(f"   {str(e)}")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
        
    except Exception as e:
        print(f"\n❌ ERROR: Pipeline failed")
        print(f"   {str(e)}")
        print(f"\nCheck the log file for details: {Config.OUTPUT_DIR / 'pipeline.log'}")
        logging.exception("Pipeline failed with exception:")
        return 1


if __name__ == '__main__':
    sys.exit(main())