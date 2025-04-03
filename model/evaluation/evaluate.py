"""
Model evaluation and reporting
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import xgboost as xgb
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)['model']
        self.test_data = pd.read_parquet("../../data/features/X_test_selected.parquet")
        self.y_test = np.load("../../data/processed/structured/y_test.npy")

    def generate_report(self):
        """Generate comprehensive evaluation report"""
        y_pred = self.model.predict(self.test_data)
        y_proba = self.model.predict_proba(self.test_data)[:, 1]

        # Metrics
        report = classification_report(self.y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(self.y_test, y_proba)

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(self.y_test, y_pred), annot=True, fmt='d')
        plt.savefig("reports/confusion_matrix.png")

        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.savefig("reports/roc_curve.png")

        # Feature Importance
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(self.model)
        plt.savefig("reports/feature_importance.png", bbox_inches='tight')

        return {
            'classification_report': report,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator("../training/saved_models/phishing_model_v1.0.joblib")
    report = evaluator.generate_report()
    print("Evaluation Complete. Reports saved to model/evaluation/reports/")