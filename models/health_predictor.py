"""
Health Risk Predictor - Predicts disease outbreak risk and healthcare demand
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthRiskPredictor:
    """Predict health risks and healthcare demand"""
    
    def __init__(self):
        self.risk_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.demand_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X: pd.DataFrame, y_risk: pd.Series, y_demand: pd.Series = None):
        """Train health prediction models"""
        logger.info("Training Health Risk Predictor...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_risk, test_size=0.2, random_state=42
        )
        
        # Train risk regression model
        self.risk_model.fit(X_train, y_train)
        y_pred = self.risk_model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics['risk_rmse'] = rmse
        self.metrics['risk_r2'] = r2
        
        logger.info(f"  Risk Model - RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Train demand classification model if target provided
        if y_demand is not None:
            X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
                X, y_demand, test_size=0.2, random_state=42
            )
            
            self.demand_model.fit(X_train_d, y_train_d)
            y_pred_d = self.demand_model.predict(X_test_d)
            
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test_d, y_pred_d)
            f1 = f1_score(y_test_d, y_pred_d, average='weighted')
            
            self.metrics['demand_accuracy'] = acc
            self.metrics['demand_f1'] = f1
            
            logger.info(f"  Demand Model - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        self.is_trained = True
        logger.info("✓ Health Risk Predictor trained successfully")
        
        return self.metrics
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict health risk scores"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.risk_model.predict(X)
    
    def predict_demand_category(self, X: pd.DataFrame) -> np.ndarray:
        """Predict demand category (Low/Medium/High)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.demand_model.predict(X)
    
    def forecast_next_days(self, recent_values: list, days: int = 7) -> list:
        """Forecast health metrics for next N days"""
        if len(recent_values) < 7:
            logger.warning("Insufficient data for accurate forecast")
            recent_values = recent_values + [recent_values[-1]] * (7 - len(recent_values))
        
        forecast = []
        current_values = recent_values[-7:]
        
        for _ in range(days):
            # Simple moving average with trend
            avg = np.mean(current_values)
            trend = (current_values[-1] - current_values[0]) / len(current_values)
            next_val = avg + trend + np.random.normal(0, avg * 0.1)
            next_val = max(0, next_val)  # No negative values
            
            forecast.append(next_val)
            current_values = current_values[1:] + [next_val]
        
        return forecast
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance = self.risk_model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str = 'models/saved_models/health_predictor.pkl'):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'risk_model': self.risk_model,
            'demand_model': self.demand_model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = 'models/saved_models/health_predictor.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.risk_model = data['risk_model']
        self.demand_model = data['demand_model']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        logger.info(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("HEALTH RISK PREDICTOR TEST")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'beds': np.random.randint(20, 500, n_samples),
        'doctors': np.random.randint(5, 100, n_samples),
        'nurses': np.random.randint(10, 200, n_samples),
        'ambulances': np.random.randint(2, 15, n_samples),
        'patients_per_day': np.random.randint(50, 800, n_samples),
        'emergency_cases': np.random.randint(5, 100, n_samples),
        'occupancy_rate': np.random.uniform(0.4, 0.95, n_samples)
    })
    
    # Create target: risk score
    y_risk = (
        X['emergency_cases'] * 0.5 +
        X['occupancy_rate'] * 50 +
        (X['patients_per_day'] / (X['doctors'] + 1)) * 0.3 +
        np.random.normal(0, 10, n_samples)
    )
    
    # Create demand category
    y_demand = pd.cut(X['patients_per_day'], bins=[0, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
    
    # Train model
    predictor = HealthRiskPredictor()
    metrics = predictor.train(X, y_risk, y_demand)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)
    print(f"Risk RMSE: {metrics['risk_rmse']:.4f}")
    print(f"Risk R²: {metrics['risk_r2']:.4f}")
    if 'demand_accuracy' in metrics:
        print(f"Demand Accuracy: {metrics['demand_accuracy']:.4f}")
        print(f"Demand F1: {metrics['demand_f1']:.4f}")
    
    # Test predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    sample = X.head(5)
    risk_predictions = predictor.predict_risk(sample)
    
    print("\nRisk Predictions:")
    for i, risk in enumerate(risk_predictions):
        print(f"  Sample {i+1}: Risk Score = {risk:.2f}")
    
    # Forecast
    print("\n" + "=" * 80)
    print("7-DAY FORECAST")
    print("=" * 80)
    
    recent_cases = [50, 52, 48, 55, 53, 51, 54]
    forecast = predictor.forecast_next_days(recent_cases, days=7)
    
    print("\nForecasted Cases:")
    for day, cases in enumerate(forecast, 1):
        print(f"  Day {day}: {cases:.0f} cases")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance = predictor.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 80)
    print("HEALTH RISK PREDICTOR READY")
    print("=" * 80)
