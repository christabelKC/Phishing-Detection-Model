"""
Real-time phishing URL detection
"""

import joblib
import pandas as pd
from urllib.parse import urlparse


class PhishingDetector:
    def __init__(self, model_path="../training/saved_models/phishing_model_v1.0.joblib"):
        # Load model and artifacts
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.feature_columns = artifacts['features']
        self.scaler = joblib.load("../features/metadata/scaler.joblib")  # Saved during training

    def extract_features(self, url: str) -> dict:
        """Extract features from raw URL"""
        parsed = urlparse(url)

        features = {
            'NumDots': url.count('.'),
            'PathLength': len(parsed.path),
            'NumDash': url.count('-'),
            'NumQueryComponents': len(parsed.query.split('&')) if parsed.query else 0,
            'HttpsInHostname': 1 if 'https' in parsed.netloc else 0,
            'IpAddress': 1 if parsed.netloc.replace('.', '').isdigit() else 0,
            'PathLevel': parsed.path.count('/'),
            # Add all other features from your original dataset
        }

        # Calculate engineered features
        features['dots_dash_ratio'] = features['NumDash'] / (features['NumDots'] + 1e-6)
        features['path_complexity'] = features['PathLevel'] * features['PathLength']

        return pd.DataFrame([features])

    def preprocess(self, url: str) -> pd.DataFrame:
        """Full preprocessing pipeline"""
        # Extract features
        raw_features = self.extract_features(url)

        # Scale features
        scaled_features = self.scaler.transform(raw_features)

        # Ensure correct feature order
        return pd.DataFrame(scaled_features, columns=self.feature_columns)

    def predict(self, url: str) -> dict:
        """Make prediction"""
        processed_data = self.preprocess(url)
        proba = self.model.predict_proba(processed_data)[0]

        return {
            'url': url,
            'phishing_probability': round(proba[1], 4),
            'is_phishing': int(proba[1] > 0.5),  # Threshold at 0.5
            'features_used': self.feature_columns
        }


if __name__ == "__main__":
    detector = PhishingDetector()

    # Example usage
    test_url = "http://suspicious-site.com/login?user=admin"
    result = detector.predict(test_url)

    print(f"URL: {result['url']}")
    print(f"Phishing Probability: {result['phishing_probability']:.2%}")
    print(f"Verdict: {'Phishing' if result['is_phishing'] else 'Legitimate'}")