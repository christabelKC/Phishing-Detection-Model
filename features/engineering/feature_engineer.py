"""
Advanced feature engineering for phishing detection
- Feature interactions
- Domain-specific transformations
- Feature scaling
"""

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)

class PhishingFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, config_path="../config/feature_config.yaml"):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.feature_columns = []

    @staticmethod
    def _load_config(config_path):
        """Load and validate feature engineering configuration"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Validate top-level sections
            required_sections = ['feature_engineering', 'feature_selection']
            for section in required_sections:
                if section not in config:
                    raise KeyError(f"Missing required section: {section}")

            # Feature Engineering validations
            fe_config = config['feature_engineering']
            fe_required = {
                'interaction_features': list,
                'scaling': bool,
                'poly_degree': int,
                'memory_optimization': bool
            }

            for key, val_type in fe_required.items():
                if key not in fe_config:
                    raise KeyError(f"Missing feature_engineering key: {key}")
                if not isinstance(fe_config[key], val_type):
                    raise TypeError(f"{key} should be {val_type.__name__}")

            if fe_config['poly_degree'] < 1:
                raise ValueError("poly_degree must be ≥ 1")

            # Feature Selection validations
            fs_config = config['feature_selection']
            fs_required = {
                'correlation_threshold': float,
                'selection_method': str,
                'xgboost_params': dict,
                'rfe_cv': int
            }

            for key, val_type in fs_required.items():
                if key not in fs_config:
                    raise KeyError(f"Missing feature_selection key: {key}")
                if not isinstance(fs_config[key], val_type):
                    raise TypeError(f"{key} should be {val_type.__name__}")

            if not 0 <= fs_config['correlation_threshold'] <= 1:
                raise ValueError("correlation_threshold must be between 0 and 1")

            valid_methods = ['xgboost', 'rfe', 'none']
            if fs_config['selection_method'] not in valid_methods:
                raise ValueError(f"selection_method must be one of {valid_methods}")

            if fs_config['rfe_cv'] < 2:
                raise ValueError("rfe_cv must be ≥ 2")

            return config

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"Config file error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise

    def _create_interaction_features(self, X):
        """Create domain-specific feature interactions"""
        # URL structure interactions
        if all(col in X.columns for col in ['NumDots', 'NumDash']):
            X['dot_dash_ratio'] = X['NumDash'] / (X['NumDots'] + 1e-6)

        # Path complexity features
        if all(col in X.columns for col in ['PathLevel', 'PathLength']):
            X['path_entropy'] = X['PathLevel'] * np.log1p(X['PathLength'])

        return X

    def _create_security_features(self, X):
        """Create security-related features"""
        # SSL/TLS features
        if 'HttpsInHostname' in X.columns:
            X['insecure_protocol'] = np.where(
                (X['HttpsInHostname'] == 0) &
                (X['IpAddress'] == 1), 1, 0
            )

        # Query parameter analysis
        if 'NumQueryComponents' in X.columns:
            X['query_frequency'] = X['NumQueryComponents'] / (X['PathLength'] + 1e-6)

        return X

    def fit(self, X, y=None):
        """Learn scaling parameters after feature engineering"""
        # Create features first
        X_engineered = self._create_interaction_features(X)
        X_engineered = self._create_security_features(X_engineered)

        # Then fit scaler on engineered features
        num_cols = X_engineered.select_dtypes(include=np.number).columns
        self.scaler.fit(X_engineered[num_cols])

        return self

    def transform(self, X):
        """Apply full feature engineering pipeline"""
        logger.info("Starting feature engineering")

        # Create advanced features
        X = self._create_interaction_features(X)
        X = self._create_security_features(X)

        # Scale numerical features
        num_cols = X.select_dtypes(include=np.number).columns
        X[num_cols] = self.scaler.transform(X[num_cols])

        # Memory optimization
        X = self._optimize_dtypes(X)

        self.feature_columns = X.columns.tolist()
        return X

    def _optimize_dtypes(self, X):
        """Optimize feature data types"""
        type_map = {
            'int64': 'int32',
            'float64': 'float32',
            'bool': 'int8'
        }

        for col in X.columns:
            current_type = str(X[col].dtype)
            if current_type in type_map:
                X[col] = X[col].astype(type_map[current_type])
        return X