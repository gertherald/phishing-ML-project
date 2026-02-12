"""
Configuration module for the phishing detection ML pipeline.
Updated based on comprehensive EDA findings.

Key Changes:
- Disabled DomainAge_Squared to avoid multicollinearity
- Added feature-specific outlier handling strategies
- Added log transformations for skewed features
- Updated model hyperparameters based on data characteristics
- Added missing value handling configuration
- Updated categorical encoding settings
"""

import os
from pathlib import Path


class Config:
    """Configuration class containing all pipeline parameters."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    DB_PATH = DATA_DIR / 'phishing.db'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    MODEL_DIR = OUTPUT_DIR / 'models'
    RESULTS_DIR = OUTPUT_DIR / 'results'
    
    # Database settings
    DB_TABLE_NAME = 'phishing_data'
    
    # Data split parameters
    TEST_SIZE = 0.2  # 20% test set
    VALIDATION_SIZE = 0.2  # 20% validation (from remaining 80%)
    RANDOM_STATE = 42
    STRATIFY = True  # Maintain 45/55 class balance
    
    # Data cleaning
    DROP_UNNAMED_COLUMNS = True  # Drop "Unnamed: 0" index column
    FIX_DUPLICATE_CATEGORIES = True  # Fix duplicate "eCommerce" in Industry
    
    # Missing value handling
    MISSING_VALUE_STRATEGY = 'median'  # For LineOfCode (22.43% missing)
    CREATE_MISSING_INDICATORS = True  # Create LineOfCode_IsMissing binary flag
    
    # Feature engineering toggles
    ENABLE_FEATURE_ENGINEERING = True
    CREATE_RATIO_FEATURES = True  # CodeComplexity, ExternalRefRatio
    CREATE_INTERACTION_FEATURES = True  # TotalReferences, TotalRedirects
    CREATE_POLYNOMIAL_FEATURES = True  # DomainAge_Squared - kept for tree models, excluded for linear models
    CREATE_BINARY_FLAGS = True  # HasExternalRefs, HasPopups, HasRedirects, HasIFrames
    CREATE_LOG_TRANSFORMS = True  # Log_NoOfImage, Log_NoOfPopup (for skewed features)
    
    # Model-specific feature handling
    USE_MODEL_SPECIFIC_FEATURES = True  # Enable different features for different models
    TREE_BASED_MODELS = ['random_forest', 'gradient_boosting', 'xgboost']
    LINEAR_MODELS = ['logistic_regression', 'svm']
    LINEAR_MODEL_EXCLUSIONS = ['DomainAge_Squared']  # Exclude from linear models (multicollinearity)
    
    # Features to protect from outlier removal (phishing signatures)
    PROTECTED_FEATURES = [
        'NoOfPopup',  # Excessive popups = phishing indicator
        'NoOfiFrame',  # Excessive iFrames = phishing indicator  
        'NoOfURLRedirect',  # Redirects = phishing technique
        'NoOfSelfRedirect'  # Self-redirects also suspicious
    ]
    
    # Features requiring extreme clipping (99th percentile) - data quality issues
    EXTREME_CLIP_FEATURES = [
        'NoOfImage'  # Extreme skewness (102.47), potential data errors
    ]
    
    # Features requiring conservative clipping (3×IQR instead of 1.5×IQR)
    CONSERVATIVE_CLIP_FEATURES = [
        'NoOfSelfRef',
        'NoOfExternalRef',
        'LargestLineLength'
    ]
    
    # Preprocessing
    HANDLE_OUTLIERS = True
    OUTLIER_METHOD = 'iqr'  # Use IQR method
    OUTLIER_ACTION = 'clip'  # Clip outliers, don't remove (preserve data)
    OUTLIER_IQR_MULTIPLIER = 1.5  # Standard IQR multiplier
    CONSERVATIVE_IQR_MULTIPLIER = 3.0  # For conservative clipping
    
    SCALING_METHOD = 'robust'  # Use RobustScaler (resistant to remaining outliers)
    
    # Categorical encoding
    CATEGORICAL_ENCODING = {
        # Binary features (Robots, IsResponsive) already 0/1 - no encoding needed
        'Industry': 'onehot',  # 11 categories after fixing duplicate eCommerce
        'HostingProvider': 'target'  # 13 categories, χ²=1739 (strongest feature)
    }
    
    # Target encoding settings
    TARGET_ENCODING_SMOOTHING = 1.0  # Smoothing to prevent overfitting
    TARGET_ENCODING_MIN_SAMPLES = 10  # Min samples for reliable encoding
    
    # Feature selection
    ENABLE_FEATURE_SELECTION = True  # Use only top 10 features
    FEATURE_SELECTION_METHOD = 'top_k'  # Use predefined top features from EDA
    TOP_K_FEATURES = 10  # Number of top features to use
    
    # Top 10 features for tree-based models (from EDA analysis)
    TOP_FEATURES_TREE = [
        'HostingProvider_Encoded',  # χ²=1739.12 (strongest)
        'HasExternalRefs',          # r=0.3814 (strongest engineered)
        'IsResponsive',             # φ=0.3298 (strongest binary)
        'DomainAgeMonths',          # r=0.3329 (strongest numerical)
        'DomainAge_Squared',        # r=0.2570 (polynomial - tree models only)
        'Robots',                   # φ=0.2413
        'TotalReferences',          # r=0.1423
        'NoOfiFrame',               # r=0.1496
        'NoOfExternalRef',          # r=0.1281
        'NoOfSelfRef'               # r=0.1144
    ]
    
    # Top 10 features for linear models (exclude DomainAge_Squared)
    TOP_FEATURES_LINEAR = [
        'HostingProvider_Encoded',  # χ²=1739.12
        'HasExternalRefs',          # r=0.3814
        'IsResponsive',             # φ=0.3298
        'DomainAgeMonths',          # r=0.3329
        'Robots',                   # φ=0.2413
        'TotalReferences',          # r=0.1423
        'NoOfiFrame',               # r=0.1496
        'NoOfExternalRef',          # r=0.1281
        'NoOfSelfRef',              # r=0.1144
        'HasPopups'                 # r=0.1178 (replaces DomainAge_Squared)
    ]
    
    # Note: 'Industry' will expand to multiple one-hot columns (Industry_*)
    # These count as 1 conceptual feature, so actual feature count may be ~15-18
    
    FEATURE_IMPORTANCE_THRESHOLD = 0.01
    CORRELATION_THRESHOLD = 0.95
    
    # Model configurations - optimized for phishing detection
    MODELS = {
        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': 500,  # Increased for better performance
                'max_depth': 20,
                'min_samples_split': 10,  # Prevent overfitting
                'min_samples_leaf': 4,  # Prevent overfitting
                'max_features': 'sqrt',
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'class_weight': 'balanced'  # Handle 1.22:1 imbalance
            },
            'tune_hyperparameters': True
        },
        'gradient_boosting': {
            'enabled': True,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.05,  # Lower for better generalization
                'max_depth': 6,
                'min_samples_split': 10,
                'min_samples_leaf': 4,
                'subsample': 0.8,  # Prevent overfitting
                'random_state': RANDOM_STATE
            },
            'tune_hyperparameters': True
        },
        'logistic_regression': {
            'enabled': True,
            'params': {
                'penalty': 'l2',
                'C': 1.0,
                'max_iter': 1000,
                'random_state': RANDOM_STATE,
                'class_weight': 'balanced',
                'solver': 'lbfgs'
            },
            'tune_hyperparameters': True
        },
        'svm': {
            'enabled': False,  # Disabled (slow on large datasets)
            'params': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'random_state': RANDOM_STATE,
                'class_weight': 'balanced',
                'probability': True
            },
            'tune_hyperparameters': False
        },
        'xgboost': {
            'enabled': True,
            'params': {
                'n_estimators': 500,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 1.22,  # Class imbalance ratio
                'random_state': RANDOM_STATE,
                'eval_metric': 'logloss',
                'n_jobs': -1
            },
            'tune_hyperparameters': True
        }
    }
    
    # Hyperparameter tuning
    CV_FOLDS = 5
    TUNING_SCORING = 'f1'  # Optimize for F1 (security context: balance precision/recall)
    TUNING_METHOD = 'grid'  # 'grid' or 'random'
    N_ITER_RANDOM_SEARCH = 50
    
    # Hyperparameter search spaces
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [300, 500, 700],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 4, 8],
            'max_features': ['sqrt', 'log2']
        },
        'gradient_boosting': {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']  # Support both penalties
        },
        'xgboost': {
            'n_estimators': [300, 500, 700],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [4, 6, 8],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    }
    
    # Evaluation
    CLASSIFICATION_THRESHOLD = 0.5  # Can be lowered to maximize recall
    EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    PRIMARY_METRIC = 'f1'  # For model selection
    GENERATE_CONFUSION_MATRIX = True
    GENERATE_CLASSIFICATION_REPORT = True
    GENERATE_ROC_CURVE = True
    GENERATE_FEATURE_IMPORTANCE = True
    
    # Logging
    VERBOSE = True
    LOG_LEVEL = 'INFO'
    SAVE_INTERMEDIATE_RESULTS = True
    
    # Model persistence
    SAVE_MODELS = True
    SAVE_PREPROCESSORS = True
    MODEL_FILE_FORMAT = 'joblib'  # 'joblib' or 'pickle'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [cls.OUTPUT_DIR, cls.MODEL_DIR, cls.RESULTS_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_enabled_models(cls):
        """Return list of enabled model names."""
        return [name for name, config in cls.MODELS.items() if config['enabled']]
    
    @classmethod
    def update_from_env(cls):
        """Update configuration from environment variables."""
        if os.getenv('TEST_SIZE'):
            cls.TEST_SIZE = float(os.getenv('TEST_SIZE'))
        if os.getenv('RANDOM_STATE'):
            cls.RANDOM_STATE = int(os.getenv('RANDOM_STATE'))
        if os.getenv('CV_FOLDS'):
            cls.CV_FOLDS = int(os.getenv('CV_FOLDS'))
        if os.getenv('SCALING_METHOD'):
            cls.SCALING_METHOD = os.getenv('SCALING_METHOD')
