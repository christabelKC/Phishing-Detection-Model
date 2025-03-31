"""
Feature selection for phishing detection
- Correlation analysis
- Model-based importance
- Recursive feature elimination
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import (RFECV, mutual_info_classif,
                                       SelectFromModel)
from xgboost import XGBClassifier
import logging

logger = logging.getLogger(__name__)


class PhishingFeatureSelector:
    def __init__(self, config_path="../config/feature_config.yaml"):
        self.config = self._load_config(config_path)
        self.selector = None
        self.selected_features = []

    @staticmethod
    def _load_config(config_path):
        """Load and validate feature selection configuration"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Validate top-level sections
            required_sections = ['feature_engineering', 'feature_selection']
            for section in required_sections:
                if section not in config:
                    raise KeyError(f"Missing required section: {section}")

            # Feature Selection validations
            fs_config = config['feature_selection']
            fs_required = {
                'correlation_threshold': float,
                'selection_method': str,
                'xgboost_params': dict
            }

            for key, val_type in fs_required.items():
                if key not in fs_config:
                    raise KeyError(f"Missing feature_selection key: {key}")
                if not isinstance(fs_config[key], val_type):
                    raise TypeError(f"{key} should be {val_type.__name__}")

            return config

        except Exception as e:
            logger.error(f"Configuration loading failed: {str(e)}")
            raise

    def correlation_analysis(self, X, y, threshold=0.9):
        """Remove highly correlated features"""
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns
                   if any(upper[column] > threshold)]

        logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        return X.drop(columns=to_drop)

    def xgboost_feature_importance(self, X, y, threshold='median'):
        """Select features using XGBoost importance"""
        xgb = XGBClassifier(
            tree_method='hist',
            enable_categorical=True,
            n_estimators=100,
            random_state=42
        )
        xgb.fit(X, y)

        selector = SelectFromModel(xgb, prefit=True, threshold=threshold)
        selected = X.columns[selector.get_support()]

        logger.info(f"Selected {len(selected)} features using XGBoost importance")
        return X[selected]

    def recursive_feature_elimination(self, X, y, cv=3):
        """Recursive Feature Elimination with CV"""
        xgb = XGBClassifier(tree_method='hist', n_estimators=50)
        selector = RFECV(xgb, step=1, cv=cv, scoring='accuracy')
        selector.fit(X, y)

        logger.info(f"Optimal number of features: {selector.n_features_}")
        return X[X.columns[selector.support_]]

    def select_features(self, X, y, method='xgboost'):
        """Main feature selection pipeline"""
        logger.info(f"Starting feature selection using {method}")

        # First remove correlated features
        X_filtered = self.correlation_analysis(X, y)

        # Apply selection method
        if method == 'xgboost':
            X_selected = self.xgboost_feature_importance(X_filtered, y)
        elif method == 'rfe':
            X_selected = self.recursive_feature_elimination(X_filtered, y)
        else:
            X_selected = X_filtered

        self.selected_features = X_selected.columns.tolist()
        return X_selected

    def save_selected_features(self, output_path):
        """Save selected feature names"""
        pd.Series(self.selected_features).to_csv(output_path, index=False)
        logger.info(f"Saved selected features to {output_path}")