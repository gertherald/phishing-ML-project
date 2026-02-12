"""
Data Preprocessor Module for Phishing Detection
Updated based on comprehensive EDA findings.

Key Updates:
- Feature-specific outlier handling (protect phishing indicators)
- Missing value handling with indicator creation
- Log transformations for skewed features
- Removed DomainAge_Squared (multicollinearity with DomainAgeMonths)
- Data cleaning (drop Unnamed: 0, fix duplicate eCommerce)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from category_encoders import TargetEncoder
import joblib
from pathlib import Path
import logging


class DataPreprocessor:
    """Handles all data preprocessing steps for phishing detection."""
    
    def __init__(self, config):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration object with preprocessing parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for fitted components
        self.scaler = None
        self.target_encoders = {}
        self.label_encoders = {}
        self.feature_names = None
        self.outlier_bounds = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data (remove duplicates, handle basic issues).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning data")
        df_clean = df.copy()
        
        # Drop unnamed index column
        if self.config.DROP_UNNAMED_COLUMNS:
            unnamed_cols = [col for col in df_clean.columns if 'Unnamed' in col]
            if unnamed_cols:
                df_clean = df_clean.drop(unnamed_cols, axis=1)
                self.logger.info(f"Dropped {len(unnamed_cols)} unnamed column(s): {unnamed_cols}")
        
        # Fix duplicate "eCommerce" category in Industry
        if self.config.FIX_DUPLICATE_CATEGORIES and 'Industry' in df_clean.columns:
            # Standardize all eCommerce variants to single value
            df_clean['Industry'] = df_clean['Industry'].replace({
                'eCommerce': 'eCommerce',
                'Ecommerce': 'eCommerce',
                'ecommerce': 'eCommerce'
            })
            self.logger.info("Fixed duplicate 'eCommerce' categories in Industry")
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dropped = initial_rows - len(df_clean)
        if dropped > 0:
            self.logger.warning(f"Removed {dropped} duplicate rows")
        
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values with strategy: median imputation + missing indicator.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the imputation strategy (True for training)
            
        Returns:
            DataFrame with imputed values
        """
        self.logger.info("Handling missing values")
        df_imputed = df.copy()
        
        # LineOfCode has 22.43% missing values (from EDA)
        if 'LineOfCode' in df.columns:
            if self.config.CREATE_MISSING_INDICATORS:
                # Create binary indicator for missingness
                df_imputed['LineOfCode_IsMissing'] = df['LineOfCode'].isnull().astype(int)
                self.logger.info(f"Created LineOfCode_IsMissing indicator: {df_imputed['LineOfCode_IsMissing'].sum()} missing values")
            
            # Impute with median (robust for skewed data)
            if fit:
                self.lineofcode_median = df['LineOfCode'].median()
                self.logger.info(f"Fitted LineOfCode median: {self.lineofcode_median}")
            
            df_imputed['LineOfCode'].fillna(self.lineofcode_median, inplace=True)
            self.logger.info(f"Imputed LineOfCode missing values with median: {self.lineofcode_median}")
        
        return df_imputed
    
    def handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle outliers using feature-specific strategies based on EDA findings.
        
        Strategy:
        - Protected features: No treatment (phishing signatures)
        - Extreme clip features: Clip to 99th percentile (data quality)
        - Conservative features: Clip to 3×IQR (preserve phishing patterns)
        - Other features: Standard 1.5×IQR clipping
        
        Args:
            df: Input DataFrame
            fit: Whether to calculate outlier bounds (True for training)
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.config.HANDLE_OUTLIERS:
            return df
        
        self.logger.info("Handling outliers with feature-specific strategies")
        df_clean = df.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        
        for col in numerical_cols:
            # Skip protected features (phishing indicators)
            if col in self.config.PROTECTED_FEATURES:
                self.logger.info(f"Protecting {col} - no outlier treatment (phishing indicator)")
                continue
            
            # Extreme clipping (99th percentile) for data quality issues
            if col in self.config.EXTREME_CLIP_FEATURES:
                if fit:
                    upper_bound = df[col].quantile(0.99)
                    self.outlier_bounds[col] = {'lower': None, 'upper': upper_bound}
                    self.logger.info(f"Extreme clip {col}: upper bound = {upper_bound:.2f} (99th percentile)")
                else:
                    upper_bound = self.outlier_bounds[col]['upper']
                
                df_clean[col] = df_clean[col].clip(upper=upper_bound)
            
            # Conservative clipping (3×IQR)
            elif col in self.config.CONSERVATIVE_CLIP_FEATURES:
                if fit:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.CONSERVATIVE_IQR_MULTIPLIER * IQR
                    upper_bound = Q3 + self.config.CONSERVATIVE_IQR_MULTIPLIER * IQR
                    self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
                    self.logger.info(f"Conservative clip {col}: [{lower_bound:.2f}, {upper_bound:.2f}] (3×IQR)")
                else:
                    lower_bound = self.outlier_bounds[col]['lower']
                    upper_bound = self.outlier_bounds[col]['upper']
                
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Standard IQR clipping (1.5×IQR) for other features
            else:
                if fit:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.OUTLIER_IQR_MULTIPLIER * IQR
                    upper_bound = Q3 + self.config.OUTLIER_IQR_MULTIPLIER * IQR
                    self.outlier_bounds[col] = {'lower': lower_bound, 'upper': upper_bound}
                else:
                    lower_bound = self.outlier_bounds[col].get('lower')
                    upper_bound = self.outlier_bounds[col].get('upper')
                
                if lower_bound is not None and upper_bound is not None:
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on EDA findings.
        
        Creates:
        - Binary flags: HasExternalRefs, HasPopups, HasRedirects, HasIFrames
        - Ratio features: CodeComplexity, ExternalRefRatio
        - Aggregation features: TotalReferences, TotalRedirects
        - Log transforms: Log_NoOfImage, Log_NoOfPopup
        
        Note: DomainAge_Squared removed to avoid multicollinearity
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if not self.config.ENABLE_FEATURE_ENGINEERING:
            return df
        
        self.logger.info("Engineering features")
        df_eng = df.copy()
        
        # Binary flags (handle skewness, top predictor: HasExternalRefs r=0.38)
        if self.config.CREATE_BINARY_FLAGS:
            df_eng['HasExternalRefs'] = (df['NoOfExternalRef'] > 0).astype(int)
            df_eng['HasPopups'] = (df['NoOfPopup'] > 0).astype(int)
            df_eng['HasRedirects'] = (df['NoOfURLRedirect'] > 0).astype(int)
            df_eng['HasIFrames'] = (df['NoOfiFrame'] > 0).astype(int)
            self.logger.info("Created 4 binary flag features")
        
        # Ratio features (capture relative patterns)
        if self.config.CREATE_RATIO_FEATURES:
            df_eng['CodeComplexity'] = df['LineOfCode'] / (df['LargestLineLength'] + 1)
            df_eng['ExternalRefRatio'] = df['NoOfExternalRef'] / (df['NoOfSelfRef'] + 1)
            self.logger.info("Created 2 ratio features")
        
        # Aggregation features
        if self.config.CREATE_INTERACTION_FEATURES:
            df_eng['TotalReferences'] = df['NoOfSelfRef'] + df['NoOfExternalRef']
            df_eng['TotalRedirects'] = df['NoOfURLRedirect'] + df['NoOfSelfRedirect']
            self.logger.info("Created 2 aggregation features")
        
        # Log transformations for skewed features (for linear models)
        if self.config.CREATE_LOG_TRANSFORMS:
            df_eng['Log_NoOfImage'] = np.log1p(df['NoOfImage'])  # log(x+1) to handle zeros
            df_eng['Log_NoOfPopup'] = np.log1p(df['NoOfPopup'])
            self.logger.info("Created 2 log-transformed features")
        
        # Polynomial features (for tree models - linear models will exclude)
        if self.config.CREATE_POLYNOMIAL_FEATURES:
            df_eng['DomainAge_Squared'] = df['DomainAgeMonths'] ** 2
            self.logger.info("Created DomainAge_Squared (will be excluded for linear models)")
        
        # Handle any infinite values from division
        df_eng.replace([np.inf, -np.inf], np.nan, inplace=True)
        if df_eng.isnull().any().any():
            df_eng.fillna(0, inplace=True)
            self.logger.warning("Replaced inf/NaN values with 0 in engineered features")
        
        return df_eng
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Strategy (based on EDA):
        - Robots, IsResponsive: Already 0/1 - keep as-is
        - Industry: One-hot encoding (11 categories)
        - HostingProvider: Target encoding (13 categories, χ²=1739 - strongest feature)
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training)
            
        Returns:
            DataFrame with encoded features
        """
        self.logger.info("Encoding categorical features")
        df_encoded = df.copy()
        
        # Get categorical columns (exclude label and binary features already encoded)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'label' in categorical_cols:
            categorical_cols.remove('label')
        
        for col in categorical_cols:
            if col not in self.config.CATEGORICAL_ENCODING:
                self.logger.warning(f"No encoding strategy specified for {col}, skipping")
                continue
            
            encoding_method = self.config.CATEGORICAL_ENCODING[col]
            
            if encoding_method == 'onehot':
                # One-hot encoding for Industry (11 categories)
                if fit:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    self.label_encoders[col] = dummies.columns.tolist()
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    self.logger.info(f"One-hot encoded {col}: {len(dummies.columns)} features")
                else:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    # Ensure same columns as training
                    expected_cols = self.label_encoders[col]
                    for exp_col in expected_cols:
                        if exp_col not in dummies.columns:
                            dummies[exp_col] = 0
                    dummies = dummies[expected_cols]
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
            
            elif encoding_method == 'target':
                # Target encoding for HostingProvider (highest χ²)
                if fit:
                    if 'label' not in df_encoded.columns:
                        raise ValueError(f"Target encoding requires 'label' column for {col}")
                    
                    # Calculate mean target per category
                    target_means = df_encoded.groupby(col)['label'].mean().to_dict()
                    self.target_encoders[col] = target_means
                    df_encoded[f'{col}_Encoded'] = df_encoded[col].map(target_means)
                    df_encoded = df_encoded.drop(col, axis=1)
                    self.logger.info(f"Target encoded {col}: {len(target_means)} categories")
                else:
                    target_means = self.target_encoders[col]
                    global_mean = np.mean(list(target_means.values()))
                    df_encoded[f'{col}_Encoded'] = df_encoded[col].map(target_means).fillna(global_mean)
                    df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using RobustScaler (resistant to outliers).
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training)
            
        Returns:
            DataFrame with scaled features
        """
        self.logger.info(f"Scaling features using {self.config.SCALING_METHOD}")
        df_scaled = df.copy()
        
        # Get numerical columns (exclude label)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'label' in numerical_cols:
            numerical_cols.remove('label')
        
        # Select scaler based on config
        if self.config.SCALING_METHOD == 'standard':
            scaler_class = StandardScaler
        elif self.config.SCALING_METHOD == 'minmax':
            scaler_class = MinMaxScaler
        elif self.config.SCALING_METHOD == 'robust':
            scaler_class = RobustScaler  # Recommended for this dataset
        else:
            self.logger.warning(f"Unknown scaling method: {self.config.SCALING_METHOD}, using RobustScaler")
            scaler_class = RobustScaler
        
        if fit:
            self.scaler = scaler_class()
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.logger.info(f"Fitted and transformed {len(numerical_cols)} numerical features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform first.")
            df_scaled[numerical_cols] = self.scaler.transform(df[numerical_cols])
            self.logger.info(f"Transformed {len(numerical_cols)} numerical features")
        
        return df_scaled
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform training data through the complete pipeline.
        
        Pipeline order:
        1. Clean data
        2. Handle missing values
        3. Handle outliers
        4. Engineer features
        5. Encode categorical features
        6. Scale features
        
        Args:
            df: Training DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("=== Fitting and transforming training data ===")
        
        # Store original feature names
        self.feature_names = df.columns.tolist()
        
        # Execute pipeline
        df = self.clean_data(df)
        df = self.handle_missing_values(df, fit=True)
        df = self.handle_outliers(df, fit=True)
        df = self.engineer_features(df)
        df = self.encode_categorical_features(df, fit=True)
        df = self.scale_features(df, fit=True)
        
        self.logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test/validation data using fitted parameters.
        
        Args:
            df: Test/validation DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("=== Transforming test/validation data ===")
        
        # Execute pipeline with fitted parameters
        df = self.clean_data(df)
        df = self.handle_missing_values(df, fit=False)
        df = self.handle_outliers(df, fit=False)
        df = self.engineer_features(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.scale_features(df, fit=False)
        
        self.logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def get_features_for_model(self, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """
        Get top 10 features for specific model type (based on EDA findings).
        
        Uses predefined top features from comprehensive EDA analysis:
        - Correlation analysis (r, φ)
        - Chi-square tests (χ²)
        - Mann-Whitney U tests
        - Domain knowledge
        
        Args:
            X: DataFrame with all features
            model_name: Name of the model (e.g., 'random_forest', 'logistic_regression')
            
        Returns:
            DataFrame with top 10 model-appropriate features
        """
        if not hasattr(self.config, 'ENABLE_FEATURE_SELECTION') or not self.config.ENABLE_FEATURE_SELECTION:
            # If feature selection disabled, use all features
            return X
        
        # Select top features based on model type
        if model_name in self.config.TREE_BASED_MODELS:
            top_features = self.config.TOP_FEATURES_TREE.copy()
            self.logger.info(f"{model_name} (tree-based): Using top 10 features (includes DomainAge_Squared)")
        else:
            top_features = self.config.TOP_FEATURES_LINEAR.copy()
            self.logger.info(f"{model_name} (linear): Using top 10 features (excludes DomainAge_Squared)")
        
        # Handle Industry one-hot encoding
        # Industry becomes multiple columns, but counts as 1 conceptual feature
        if 'Industry' in top_features:
            industry_cols = [col for col in X.columns if col.startswith('Industry_')]
            if industry_cols:
                # Remove 'Industry' placeholder, add actual one-hot columns
                top_features.remove('Industry')
                top_features.extend(industry_cols)
                self.logger.info(f"  Expanded Industry into {len(industry_cols)} one-hot features")
        
        # Select features that exist in X
        available_features = [f for f in top_features if f in X.columns]
        missing_features = [f for f in top_features if f not in X.columns]
        
        if missing_features:
            self.logger.warning(f"  Missing features (not yet created): {missing_features}")
        
        # Filter DataFrame
        X_selected = X[available_features].copy()
        
        self.logger.info(f"  Features selected: {len(available_features)} (from {X.shape[1]} total)")
        self.logger.info(f"  Selected features: {available_features}")
        
        return X_selected
    
    def split_features_target(self, df: pd.DataFrame):
        """
        Split features and target variable.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y)
        """
        if 'label' not in df.columns:
            raise ValueError("Target column 'label' not found in DataFrame")
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        self.logger.info(f"Split features and target: X shape = {X.shape}, y shape = {y.shape}")
        return X, y
    
    def save(self, filepath: Path):
        """Save the fitted preprocessor."""
        preprocessor_state = {
            'scaler': self.scaler,
            'target_encoders': self.target_encoders,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'outlier_bounds': self.outlier_bounds,
            'lineofcode_median': getattr(self, 'lineofcode_median', None)
        }
        joblib.dump(preprocessor_state, filepath)
        self.logger.info(f"Saved preprocessor to {filepath}")
    
    def load(self, filepath: Path):
        """Load a fitted preprocessor."""
        preprocessor_state = joblib.load(filepath)
        self.scaler = preprocessor_state['scaler']
        self.target_encoders = preprocessor_state['target_encoders']
        self.label_encoders = preprocessor_state['label_encoders']
        self.feature_names = preprocessor_state['feature_names']
        self.outlier_bounds = preprocessor_state['outlier_bounds']
        self.lineofcode_median = preprocessor_state.get('lineofcode_median')
        self.logger.info(f"Loaded preprocessor from {filepath}")
