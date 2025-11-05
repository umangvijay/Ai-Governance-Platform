"""
Infrastructure Failure Predictor - Predicts infrastructure failures and maintenance needs
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfrastructureFailurePredictor:
    """Predict infrastructure failures using XGBoost"""
    
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        self.is_trained = False
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train infrastructure failure prediction model"""
        logger.info("Training Infrastructure Failure Predictor...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        self.metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"  Accuracy: {acc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        self.is_trained = True
        logger.info("✓ Infrastructure Failure Predictor trained successfully")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure (0=No, 1=Yes)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure probability"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def predict_high_risk_locations(self, X: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
        """Identify high-risk locations"""
        probas = self.predict_proba(X)
        failure_proba = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        
        high_risk = failure_proba >= threshold
        
        results = pd.DataFrame({
            'index': X.index,
            'failure_probability': failure_proba,
            'high_risk': high_risk
        })
        
        return results[results['high_risk']].sort_values('failure_probability', ascending=False)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str = 'models/saved_models/infrastructure_predictor.pkl'):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = 'models/saved_models/infrastructure_predictor.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        logger.info(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("INFRASTRUCTURE FAILURE PREDICTOR TEST")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'response_time_hours': np.random.uniform(1, 72, n_samples),
        'resolution_time_hours': np.random.uniform(2, 168, n_samples),
        'citizen_satisfaction': np.random.randint(1, 6, n_samples),
        'recurring': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'priority_numeric': np.random.randint(0, 4, n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'month': np.random.randint(1, 13, n_samples),
    })
    
    # Create target: failure (based on conditions)
    y = (
        (X['response_time_hours'] > 48) |
        (X['recurring'] == 1) |
        (X['citizen_satisfaction'] < 3)
    ).astype(int)
    
    # Train model
    predictor = InfrastructureFailurePredictor()
    metrics = predictor.train(X, y)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Test predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    sample = X.head(10)
    predictions = predictor.predict(sample)
    probabilities = predictor.predict_proba(sample)
    
    print("\nFailure Predictions:")
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        failure_prob = proba[1] if len(proba) > 1 else proba[0]
        print(f"  Sample {i+1}: {'FAILURE' if pred == 1 else 'NO FAILURE'} (Probability: {failure_prob:.2%})")
    
    # High risk locations
    print("\n" + "=" * 80)
    print("HIGH RISK LOCATIONS")
    print("=" * 80)
    
    high_risk = predictor.predict_high_risk_locations(X, threshold=0.7)
    print(f"\nFound {len(high_risk)} high-risk locations:")
    print(high_risk.head(10).to_string(index=False))
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance = predictor.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 80)
    print("INFRASTRUCTURE FAILURE PREDICTOR READY")
    print("=" * 80)
