"""
Train the missing Health and Infrastructure models with synthetic data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from models.health_predictor import HealthRiskPredictor
from models.infrastructure_predictor import InfrastructureFailurePredictor

print("=" * 80)
print("TRAINING MISSING MODELS (Health & Infrastructure)")
print("=" * 80)

try:
    # Train Health Risk Predictor
    print("\n[1/2] Training Health Risk Predictor...")
    np.random.seed(42)
    n_samples = 500
    
    # Create synthetic health data
    X_health = pd.DataFrame({
        'beds': np.random.randint(20, 500, n_samples),
        'doctors': np.random.randint(5, 100, n_samples),
        'nurses': np.random.randint(10, 200, n_samples),
        'ambulances': np.random.randint(2, 15, n_samples),
        'patients_per_day': np.random.randint(50, 800, n_samples),
        'emergency_cases': np.random.randint(5, 100, n_samples),
        'occupancy_rate': np.random.uniform(0.4, 0.95, n_samples)
    })
    
    # Create target
    y_health_risk = (
        X_health['emergency_cases'] * 0.5 +
        X_health['occupancy_rate'] * 50 +
        (X_health['patients_per_day'] / (X_health['doctors'] + 1)) * 0.3 +
        np.random.normal(0, 10, n_samples)
    )
    
    y_health_demand = pd.cut(X_health['patients_per_day'], bins=[0, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
    
    # Train
    health_model = HealthRiskPredictor()
    health_model.train(X_health, y_health_risk, y_health_demand)
    health_model.save_model()
    print(">> Health Risk Predictor trained and saved!")
    
    # Train Infrastructure Predictor
    print("\n[2/2] Training Infrastructure Failure Predictor...")
    
    # Create synthetic infrastructure data
    X_infra = pd.DataFrame({
        'response_time_hours': np.random.uniform(1, 120, n_samples),
        'resolution_time_hours': np.random.uniform(2, 240, n_samples),
        'citizen_satisfaction': np.random.randint(1, 6, n_samples),
        'recurring': np.random.randint(0, 2, n_samples),
        'priority_numeric': np.random.randint(1, 6, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    })
    
    # Create target
    y_infra = (
        (X_infra['response_time_hours'] > 48) |
        (X_infra['recurring'] == 1) |
        (X_infra['citizen_satisfaction'] < 3)
    ).astype(int)
    
    # Train
    infra_model = InfrastructureFailurePredictor()
    infra_model.train(X_infra, y_infra)
    infra_model.save_model()
    print(">> Infrastructure Failure Predictor trained and saved!")
    
    print("\n" + "=" * 80)
    print(">> ALL MISSING MODELS TRAINED!")
    print("=" * 80)
    print("\nNow all 4 models are ready:")
    print("  - Health Risk Predictor")
    print("  - Infrastructure Failure Predictor")
    print("  - Demand Forecaster")
    print("  - Sentiment Analyzer")
    print("\nRestart Flask app: python app.py")
    print("=" * 80)
    
except Exception as e:
    print(f"\n!! Error: {e}")
    import traceback
    traceback.print_exc()
