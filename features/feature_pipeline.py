# features/feature_pipeline.py
import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Tuple
from engineering.feature_engineer import PhishingFeatureEngineer
from selection.feature_selector import PhishingFeatureSelector

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, config_path: str = "config/feature_config.yaml"):
        self.config = self._load_config(config_path)
        self.engineer = PhishingFeatureEngineer(config_path)
        self.selector = PhishingFeatureSelector(config_path)
        self.feature_metadata = {}

    def _load_config(self, config_path: str) -> dict:
        """Load and validate pipeline configuration"""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def run_engineering_pipeline(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Execute full feature engineering pipeline"""
        try:
            if fit:
                X_engineered = self.engineer.fit_transform(X)
                self.feature_metadata['engineered_columns'] = X_engineered.columns.tolist()
            else:
                X_engineered = self.engineer.transform(X)

            logger.info(f"Engineered features shape: {X_engineered.shape}")
            return X_engineered

        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise

    def run_selection_pipeline(self, X: pd.DataFrame, y: pd.Series, fit: bool = True) -> pd.DataFrame:
        """Execute full feature selection pipeline"""
        try:
            if fit:
                X_selected = self.selector.select_features(X, y)
                self.feature_metadata['selected_columns'] = X_selected.columns.tolist()
            else:
                X_selected = X[self.selector.selected_features]

            logger.info(f"Selected features shape: {X_selected.shape}")
            return X_selected

        except Exception as e:
            logger.error(f"Feature selection failed: {str(e)}")
            raise

    def execute_full_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """End-to-end feature transformation pipeline"""
        # Feature Engineering
        logger.info("Starting feature engineering pipeline")
        X_train_eng = self.run_engineering_pipeline(X_train, fit=True)
        X_test_eng = self.run_engineering_pipeline(X_test, fit=False) if X_test is not None else None

        # Feature Selection
        logger.info("Starting feature selection pipeline")
        X_train_sel = self.run_selection_pipeline(X_train_eng, y_train, fit=True)
        X_test_sel = self.run_selection_pipeline(X_test_eng, y_train, fit=False) if X_test is not None else None

        return X_train_sel, X_test_sel

    def save_artifacts(self, output_dir: str):
        """Save pipeline artifacts"""
        os.makedirs(output_dir, exist_ok=True)

        # Save engineered features metadata
        pd.Series(self.feature_metadata.get('engineered_columns', [])).to_csv(
            os.path.join(output_dir, 'engineered_features.csv'),
            index=False
        )

        # Save selected features metadata
        pd.Series(self.feature_metadata.get('selected_columns', [])).to_csv(
            os.path.join(output_dir, 'selected_features.csv'),
            index=False
        )

        logger.info(f"Saved feature artifacts to {output_dir}")


# features/feature_pipeline.py

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load processed data
    data_dir = "../data/processed/structured/"

    # Load features from parquet
    X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(data_dir, "X_test.parquet"))

    # Load targets from numpy arrays
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))  # If available

    # Convert to pandas Series for compatibility
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test) if os.path.exists(os.path.join(data_dir, "y_test.npy")) else None

    # Initialize pipeline
    pipeline = FeaturePipeline()

    # Execute full pipeline
    X_train_final, X_test_final = pipeline.execute_full_pipeline(X_train, y_train, X_test)

    # Save processed features
    feature_dir = "../data/features/"
    X_train_final.to_parquet(os.path.join(feature_dir, "X_train_selected.parquet"))

    if X_test_final is not None:
        X_test_final.to_parquet(os.path.join(feature_dir, "X_test_selected.parquet"))

    # Save pipeline artifacts
    pipeline.save_artifacts(os.path.join(feature_dir, "metadata"))

if __name__ == "__main__":
    main()