"""
Data Loader Module

WHAT IT DOES:
- Connects to SQLite database and loads phishing dataset
- Performs stratified train/validation/test split
- Maintains class balance across splits

WHY WE NEED IT:
- Separates data loading from preprocessing
- Reusable for both training and inference
- Ensures reproducible data splits with stratification
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging


class DataLoader:
    """Handles loading data from database and splitting."""
    
    def __init__(self, config):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object with database settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = config.DB_PATH
        self.table_name = config.DB_TABLE_NAME
    
    def load_data(self):
        """
        Load data from SQLite database.
        
        Returns:
            pd.DataFrame: Complete dataset with all features and target
        
        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If table doesn't exist in database
        """
        self.logger.info(f"Loading data from {self.db_path}")
        
        # Check database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Load table
            query = f"SELECT * FROM {self.table_name}"
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            self.logger.info(f"Loaded {len(df)} records from database")
            self.logger.info(f"Features: {df.shape[1]} columns")
            self.logger.info(f"Target distribution:\n{df['label'].value_counts()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets with stratification.
        
        Strategy:
        - 60% training
        - 20% validation
        - 20% test
        - Stratified to maintain class balance (45% phishing, 55% legitimate)
        
        Args:
            df: Complete DataFrame
            
        Returns:
            Tuple: (train_df, val_df, test_df)
        """
        self.logger.info("Splitting data into train/val/test sets")
        
        # Check if label column exists
        if 'label' not in df.columns:
            raise ValueError("Target column 'label' not found in DataFrame")
        
        # First split: 80% train+val, 20% test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=df['label'] if self.config.STRATIFY else None
        )
        
        # Second split: 75% of train_val (60% overall) train, 25% of train_val (20% overall) val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE),  # 0.25
            random_state=self.config.RANDOM_STATE,
            stratify=train_val_df['label'] if self.config.STRATIFY else None
        )
        
        # Log split statistics
        self.logger.info(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Class distribution: {train_df['label'].value_counts().to_dict()}")
        
        self.logger.info(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Class distribution: {val_df['label'].value_counts().to_dict()}")
        
        self.logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        self.logger.info(f"  Class distribution: {test_df['label'].value_counts().to_dict()}")
        
        return train_df, val_df, test_df
    
    def get_feature_names(self, df):
        """Get list of feature names (excluding target)."""
        features = [col for col in df.columns if col != 'label']
        return features
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the dataset.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_samples': len(df),
            'n_features': df.shape[1] - 1,  # Excluding label
            'n_phishing': (df['label'] == 0).sum(),
            'n_legitimate': (df['label'] == 1).sum(),
            'phishing_percentage': (df['label'] == 0).sum() / len(df) * 100,
            'legitimate_percentage': (df['label'] == 1).sum() / len(df) * 100,
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return summary
