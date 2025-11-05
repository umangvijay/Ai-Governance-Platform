"""
Service Demand Forecaster - Predicts future service demand
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """Forecast service demand using time series features"""
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        self.is_trained = False
        self.feature_names = None
        self.metrics = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """Train demand forecasting model"""
        logger.info("Training Service Demand Forecaster...")
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        
        self.is_trained = True
        logger.info("✓ Demand Forecaster trained successfully")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict service demand"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)  # No negative demand
    
    def forecast_next_hours(self, recent_data: pd.DataFrame, hours: int = 24) -> list:
        """Forecast demand for next N hours"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        forecasts = []
        
        # Use last data point as template
        last_row = recent_data.iloc[-1].copy()
        
        for h in range(hours):
            # Update time features
            current_hour = (int(last_row.get('hour', 0)) + h) % 24
            current_day = int(last_row.get('day_of_week', 0))
            
            if current_hour == 0 and h > 0:
                current_day = (current_day + 1) % 7
            
            # Create prediction features
            pred_features = last_row.copy()
            pred_features['hour'] = current_hour
            pred_features['day_of_week'] = current_day
            pred_features['is_business_hours'] = 1 if 9 <= current_hour <= 17 else 0
            pred_features['is_night'] = 1 if current_hour in [22, 23, 0, 1, 2, 3, 4, 5] else 0
            
            # Predict
            X_pred = pd.DataFrame([pred_features])[self.feature_names]
            demand = self.predict(X_pred)[0]
            
            forecasts.append({
                'hour': h + 1,
                'time_of_day': current_hour,
                'predicted_demand': int(demand)
            })
        
        return forecasts
    
    def identify_peak_hours(self, forecasts: list) -> list:
        """Identify peak demand hours"""
        df = pd.DataFrame(forecasts)
        peak_threshold = df['predicted_demand'].quantile(0.75)
        
        peaks = df[df['predicted_demand'] >= peak_threshold].to_dict('records')
        return sorted(peaks, key=lambda x: x['predicted_demand'], reverse=True)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save_model(self, path: str = 'models/saved_models/demand_forecaster.pkl'):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = 'models/saved_models/demand_forecaster.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        logger.info(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("DEMAND FORECASTER TEST")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 2000
    
    # Time-based features
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)
    
    X = pd.DataFrame({
        'hour': hours,
        'day_of_week': days,
        'month': np.random.randint(1, 13, n_samples),
        'is_business_hours': (hours >= 9) & (hours <= 17),
        'is_night': np.isin(hours, [22, 23, 0, 1, 2, 3, 4, 5]),
        'service_type_encoded': np.random.randint(0, 5, n_samples),
        'urgency': np.random.randint(1, 11, n_samples)
    })
    
    # Create target: demand based on time patterns
    base_demand = 50
    y = (
        base_demand +
        (X['is_business_hours'] * 30) +
        (X['urgency'] * 5) +
        (X['day_of_week'].isin([0, 1, 2, 3, 4]) * 20) +
        np.random.normal(0, 10, n_samples)
    )
    y = np.maximum(y, 0)
    
    # Train model
    forecaster = DemandForecaster()
    metrics = forecaster.train(X, y)
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Test predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    sample = X.head(5)
    predictions = forecaster.predict(sample)
    
    print("\nDemand Predictions:")
    for i, pred in enumerate(predictions):
        print(f"  Sample {i+1}: {int(pred)} requests")
    
    # Forecast next 24 hours
    print("\n" + "=" * 80)
    print("24-HOUR FORECAST")
    print("=" * 80)
    
    recent_data = X.tail(1)
    forecast = forecaster.forecast_next_hours(recent_data, hours=24)
    
    print("\nHourly Forecast:")
    for item in forecast[:12]:  # Show first 12 hours
        print(f"  Hour {item['hour']:2d} (Time: {item['time_of_day']:02d}:00) - {item['predicted_demand']} requests")
    
    # Peak hours
    print("\n" + "=" * 80)
    print("PEAK DEMAND HOURS")
    print("=" * 80)
    
    peaks = forecaster.identify_peak_hours(forecast)
    print(f"\nTop 5 Peak Hours:")
    for item in peaks[:5]:
        print(f"  Hour {item['hour']:2d}: {item['predicted_demand']} requests")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance = forecaster.get_feature_importance()
    print(importance.to_string(index=False))
    
    # Save model
    forecaster.save_model()
    
    print("\n" + "=" * 80)
    print("DEMAND FORECASTER READY")
    print("=" * 80)
