"""
XGBoost training pipeline for phishing detection
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhishingModelTrainer:
    def __init__(self, config_path="config/training_config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = []

    def _load_config(self, config_path):
        """Load training configuration"""
        with open(config_path) as f:
            return yaml.safe_load(f)

    def load_data(self):
        """Load feature-engineered data"""
        logger.info("Loading training data")
        try:
            X_train = pd.read_parquet("../../data/features/X_train_selected.parquet")
            y_train = np.load("../../data/processed/structured/y_train.npy")
            return X_train, y_train
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def initialize_model(self):
        """Create XGBoost classifier with optimized params"""
        logger.info("Initializing model")
        return xgb.XGBClassifier(
            objective='binary:logistic',
            tree_method='hist',
            enable_categorical=True,
            **self.config['model_params']  # Now includes early_stopping_rounds
        )

    def train(self):
        """Full training pipeline"""
        try:
            # Load data
            X_train, y_train = self.load_data()
            self.feature_columns = X_train.columns.tolist()

            # Initialize model
            self.model = self.initialize_model()

            # Train with validation split
            logger.info("Starting training")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=self.config['validation_size'],
                stratify=y_train
            )

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True  # Removed early_stopping_rounds from here
            )

            # Save model
            self.save_model()
            return self.model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def save_model(self):
        """Save trained model and metadata"""
        os.makedirs("saved_models", exist_ok=True)
        model_path = f"saved_models/phishing_model_v{self.config['model_version']}.joblib"
        joblib.dump({
            'model': self.model,
            'features': self.feature_columns,
            'config': self.config
        }, model_path)
        logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    trainer = PhishingModelTrainer()
    model = trainer.train()