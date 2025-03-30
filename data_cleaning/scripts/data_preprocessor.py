import pandas as pd
import numpy as np
import logging
import os
import yaml
import pyarrow
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

"""
Data Preprocessing Module
This module is responsible for loading, cleaning, and preprocessing the dataset.
It includes functions for handling missing values, outlier detection,
feature engineering, and data splitting.
"""
class DataPreprocessor:
    def __init__(self, config_path="../../config/preprocessing_config.yaml"):
        self.config = self._load_config(config_path)
        self.encoders = {}
        self.feature_columns = []

    @staticmethod
    def _load_config(config_path):
        """Load and validate preprocessing configuration"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Validate required keys
            required_keys = [
                'missing_values_threshold',
                'skew_threshold',
                'max_onehot_categories',
                'test_size',
                'random_state'
            ]

            for key in required_keys:
                if key not in config:
                    raise KeyError(f"Missing required configuration key: {key}")

            # Additional config validations
            assert 0 < config['missing_values_threshold'] < 1, "Invalid missing values threshold"
            assert 0 < config['test_size'] < 0.5, "Invalid test size"

            return config

        except (yaml.YAMLError, IOError) as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def _memory_optimize(self, data):
        """ Optimize memory usage of dataframe """
        for col in data.columns:
            if data[col].dtype == "int64":
                data[col] = pd.to_numeric(data[col], downcast="integer")
            elif data[col].dtype == "float64":
                data[col] = pd.to_numeric(data[col], downcast="float")
        return data

    def load_raw_data(self, path):
        """Load and optimize raw dataset with enhanced error handling"""
        try:
            logger.info(f"Loading data from {path}")

            if not os.path.exists(path):
                raise FileNotFoundError(f"Input file not found: {path}")

            # Existing loading logic
            if path.endswith('.parquet'):
                data = pd.read_parquet(path)
            elif path.endswith('.csv'):
                data = pd.read_csv(path, low_memory=False)
            else:
                raise ValueError("Unsupported file format")

            # Additional data validation
            if data.empty:
                raise ValueError("Loaded dataset is empty")

            return self._memory_optimize(data)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _clean_data(self, data):
        """Data cleaning pipeline - corrected"""
        logger.info("Cleaning data")

        # Validate target exists
        if 'Phising' not in data.columns:
            raise ValueError("Target column 'is_phishing' missing")

        # Drop columns with >50% missing values (keep target)
        missing_threshold = self.config['missing_values_threshold']
        cols_to_keep = [col for col in data.columns
                        if data[col].isnull().mean() < missing_threshold or col == 'is_phishing']

        data = data[cols_to_keep]

        # Impute remaining missing values (except target)
        for col in data.columns:
            if col != 'is_phishing' and data[col].isnull().any():
                if data[col].dtype == 'object':
                    data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    data[col] = data[col].fillna(data[col].median())

        return self._memory_optimize(data)

    def detect_and_handle_outliers(self, data):
        """Advanced outlier detection and handling"""
        logger.info("Detecting and handling outliers")

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            # Inter quartile Range (IQR) method for outlier detection
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count and log outliers
            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            logger.info(f"Outliers in {col}: {len(outliers)} rows")

            # Factorizations: cap at bounds instead of removing
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound

        return self._memory_optimize(data)

    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create phishing-specific features from numerical data"""

        # Drop index column if present
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

        # Create interaction features
        data['dots_dash_ratio'] = data['NumDash'] / (data['NumDots'] + 1e-6)
        data['path_complexity'] = data['PathLevel'] * data['PathLength']

        # Create aggregated security features
        data['total_special_chars'] = (
                data['NumDots'] +
                data['NumDash'] +
                data['NumPercent']
        )

        # Create URL structure flags
        data['suspicious_path'] = np.where(
            (data['PathLength'] > 50) | (data['PathLevel'] > 5), 1, 0
        )

        # Create query complexity metric
        data['query_density'] = data['NumQueryComponents'] / (data['PathLength'] + 1e-6)

        # IP address features
        data['ip_in_hostname'] = np.where(
            (data['IpAddress'] == 1) & (data['HttpsInHostname'] == 0), 1, 0
        )

        # Log-transform skewed features
        skewed_features = ['PathLength', 'NumNumericChars']
        for feat in skewed_features:
            if data[feat].skew() > 1.0:
                data[f'log_{feat}'] = np.log1p(data[feat])

        # Memory optimization
        data = data.astype({
            'suspicious_path': 'int8',
            'ip_in_hostname': 'int8',
            'dots_dash_ratio': 'float32',
            'path_complexity': 'int16'
        })

        return data

    def split_and_save(self, X, y, output_dir):
        """Optimized splitting for numerical data"""
        # Use stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            stratify=y,
            random_state=self.config['random_state']
        )

        # Save with explicit engine specification
        X_train.to_parquet(
            os.path.join(output_dir, 'X_train.parquet'),
            engine='pyarrow'  # or 'fastparquet'
        )
        X_test.to_parquet(
            os.path.join(output_dir, 'X_test.parquet'),
            engine='pyarrow'  # or 'fastparquet'
        )

        # Save targets
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train.values)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test.values)

    def run_pipeline(self, input_path, output_dir):
        """Execute complete preprocessing pipeline"""
        try:
            # Load and process data
            data = self.load_raw_data(input_path)
            data = self._clean_data(data)
            data = self._engineer_features(data)

            # Handle target separation directly (no encoding needed)
            # Assuming your target column is named 'is_phishing'
            X = data.drop(columns=['Phising'])
            y = data['Phising']

            # Split and save data
            self.split_and_save(X, y, output_dir)

            logger.info("Preprocessing completed successfully")
            return True

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    preprocessor.run_pipeline(
        input_path="../../data/raw/Phising_dataset_predict.csv",
        output_dir="../../data/processed/structured/"
    )