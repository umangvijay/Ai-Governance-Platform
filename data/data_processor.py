"""
Data Processor - Clean and prepare data for ML models
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and prepare data for ML training"""
    
    def __init__(self, raw_data: Dict[str, pd.DataFrame]):
        self.raw_data = raw_data
        self.processed_data = {}
        
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Process all datasets"""
        logger.info("=" * 80)
        logger.info("PROCESSING DATASETS FOR ML")
        logger.info("=" * 80)
        
        # Process health data
        if self._has_health_data():
            self.processed_data['health'] = self._process_health_data()
            logger.info("✓ Health data processed")
        
        # Process infrastructure complaints
        if self._has_infrastructure_data():
            self.processed_data['infrastructure'] = self._process_infrastructure_data()
            logger.info("✓ Infrastructure data processed")
        
        # Process service requests
        if self._has_service_requests():
            self.processed_data['service_requests'] = self._process_service_requests()
            logger.info("✓ Service requests processed")
        
        # Process citizen feedback
        if self._has_feedback_data():
            self.processed_data['citizen_feedback'] = self._process_feedback_data()
            logger.info("✓ Citizen feedback processed")
        
        logger.info(f"\n✓ Total processed datasets: {len(self.processed_data)}")
        return self.processed_data
    
    def _has_health_data(self) -> bool:
        return any('health' in key or 'hospital' in key for key in self.raw_data.keys())
    
    def _has_infrastructure_data(self) -> bool:
        return any('complaint' in key or 'infrastructure' in key for key in self.raw_data.keys())
    
    def _has_service_requests(self) -> bool:
        return 'service_requests' in self.raw_data
    
    def _has_feedback_data(self) -> bool:
        return any('feedback' in key or 'sentiment' in key for key in self.raw_data.keys())
    
    def _process_health_data(self) -> pd.DataFrame:
        """Process health infrastructure data"""
        # Find health dataset
        health_key = next((k for k in self.raw_data if 'health' in k or 'hospital' in k), None)
        if not health_key:
            return pd.DataFrame()
        
        df = self.raw_data[health_key].copy()
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        # Calculate derived features
        if 'beds' in df.columns and 'patients_per_day' in df.columns:
            df['bed_utilization'] = df['patients_per_day'] / (df['beds'] + 1)
        
        if 'doctors' in df.columns and 'patients_per_day' in df.columns:
            df['patients_per_doctor'] = df['patients_per_day'] / (df['doctors'] + 1)
        
        # Create risk score
        df['risk_score'] = 0.0
        if 'emergency_cases' in df.columns:
            df['risk_score'] += (df['emergency_cases'] - df['emergency_cases'].mean()) / df['emergency_cases'].std()
        if 'occupancy_rate' in df.columns:
            df['risk_score'] += (df['occupancy_rate'] - 0.5) * 2
        
        return df
    
    def _process_infrastructure_data(self) -> pd.DataFrame:
        """Process infrastructure complaints"""
        # Find infrastructure dataset
        infra_key = next((k for k in self.raw_data if 'complaint' in k or 'infrastructure' in k), None)
        if not infra_key:
            return pd.DataFrame()
        
        df = self.raw_data[infra_key].copy()
        
        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables
        if 'issue_type' in df.columns:
            df['issue_type_encoded'] = pd.Categorical(df['issue_type']).codes
        
        if 'status' in df.columns:
            df['status_encoded'] = pd.Categorical(df['status']).codes
        
        if 'priority' in df.columns:
            priority_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
            df['priority_numeric'] = df['priority'].map(priority_map).fillna(1)
        
        # Calculate failure risk
        df['failure_risk'] = 0.0
        if 'recurring' in df.columns:
            df['failure_risk'] += df['recurring'] * 0.3
        if 'response_time_hours' in df.columns:
            df['failure_risk'] += (df['response_time_hours'] > 24).astype(int) * 0.4
        if 'citizen_satisfaction' in df.columns:
            df['failure_risk'] += (df['citizen_satisfaction'] < 3).astype(int) * 0.3
        
        return df
    
    def _process_service_requests(self) -> pd.DataFrame:
        """Process service requests"""
        df = self.raw_data['service_requests'].copy()
        
        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
        
        # Encode service types
        if 'service_type' in df.columns:
            df['service_type_encoded'] = pd.Categorical(df['service_type']).codes
        
        # Time-based features
        if 'hour' in df.columns:
            df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
            df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        return df
    
    def _process_feedback_data(self) -> pd.DataFrame:
        """Process citizen feedback"""
        # Find feedback dataset
        feedback_key = next((k for k in self.raw_data if 'feedback' in k or 'sentiment' in k), None)
        if not feedback_key:
            return pd.DataFrame()
        
        df = self.raw_data[feedback_key].copy()
        
        # Encode sentiment
        if 'sentiment' in df.columns:
            sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
            df['sentiment_numeric'] = df['sentiment'].map(sentiment_map).fillna(1)
        
        # Rating normalization
        if 'rating' in df.columns:
            df['rating_normalized'] = (df['rating'] - 1) / 4  # 0 to 1 scale
        
        # Convert dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    def get_ml_ready_data(self, dataset_name: str, feature_cols: list, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Get features and target for ML"""
        if dataset_name not in self.processed_data:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.processed_data[dataset_name]
        
        # Filter available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            raise ValueError(f"No features found in dataset {dataset_name}")
        
        X = df[available_features].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)
        
        if y is not None:
            y = y.fillna(y.median() if y.dtype != 'object' else y.mode()[0])
        
        return X, y
    
    def create_time_series_data(self, dataset_name: str, target_col: str, window_size: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """Create time series data for forecasting"""
        if dataset_name not in self.processed_data:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        df = self.processed_data[dataset_name]
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        values = df[target_col].values
        
        X, y = [], []
        for i in range(len(values) - window_size):
            X.append(values[i:i+window_size])
            y.append(values[i+window_size])
        
        return np.array(X), np.array(y)


if __name__ == "__main__":
    from data_loader import RealDataLoader
    
    print("=" * 80)
    print("DATA PROCESSOR TEST")
    print("=" * 80)
    
    # Load data
    loader = RealDataLoader()
    raw_data = loader.load_all_datasets()
    
    # Process data
    processor = DataProcessor(raw_data)
    processed_data = processor.process_all()
    
    print("\n" + "=" * 80)
    print("PROCESSED DATA SUMMARY")
    print("=" * 80)
    
    for name, df in processed_data.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)[:10]}")  # First 10 columns
        print(f"  Sample:\n{df.head(3)}")
    
    print("\n" + "=" * 80)
    print("DATA PROCESSOR READY")
    print("=" * 80)
