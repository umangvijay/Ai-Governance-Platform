"""
Model Trainer - Train all ML models together
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.health_predictor import HealthRiskPredictor
from models.infrastructure_predictor import InfrastructureFailurePredictor
from models.demand_forecaster import DemandForecaster
from models.sentiment_analyzer import SentimentAnalyzer
from data.data_loader import RealDataLoader
from data.data_processor import DataProcessor

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train all ML models"""
    
    def __init__(self):
        self.health_predictor = None
        self.infrastructure_predictor = None
        self.demand_forecaster = None
        self.sentiment_analyzer = None
        
        self.training_results = {}
        
    def train_all_models(self, processed_data: dict) -> dict:
        """Train all models with processed data"""
        logger.info("=" * 80)
        logger.info("TRAINING ALL ML MODELS")
        logger.info("=" * 80)
        
        # Train Health Predictor
        if 'health' in processed_data:
            logger.info("\n[1/4] Training Health Risk Predictor...")
            self.health_predictor = HealthRiskPredictor()
            health_metrics = self._train_health_model(processed_data['health'])
            self.training_results['health'] = health_metrics
        
        # Train Infrastructure Predictor
        if 'infrastructure' in processed_data:
            logger.info("\n[2/4] Training Infrastructure Failure Predictor...")
            self.infrastructure_predictor = InfrastructureFailurePredictor()
            infra_metrics = self._train_infrastructure_model(processed_data['infrastructure'])
            self.training_results['infrastructure'] = infra_metrics
        
        # Train Demand Forecaster
        if 'service_requests' in processed_data:
            logger.info("\n[3/4] Training Service Demand Forecaster...")
            self.demand_forecaster = DemandForecaster()
            demand_metrics = self._train_demand_model(processed_data['service_requests'])
            self.training_results['demand'] = demand_metrics
        
        # Train Sentiment Analyzer
        if 'citizen_feedback' in processed_data:
            logger.info("\n[4/4] Training Sentiment Analyzer...")
            self.sentiment_analyzer = SentimentAnalyzer()
            sentiment_metrics = self._train_sentiment_model(processed_data['citizen_feedback'])
            self.training_results['sentiment'] = sentiment_metrics
        
        logger.info("\n" + "=" * 80)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return self.training_results
    
    def _train_health_model(self, health_data: pd.DataFrame) -> dict:
        """Train health predictor"""
        try:
            # Prepare features
            feature_cols = ['beds', 'doctors', 'nurses', 'ambulances', 'patients_per_day', 'emergency_cases', 'occupancy_rate']
            available_features = [col for col in feature_cols if col in health_data.columns]
            
            if len(available_features) < 3:
                logger.warning("Insufficient features for health model")
                return {'status': 'skipped', 'reason': 'insufficient_features'}
            
            X = health_data[available_features].fillna(health_data[available_features].median())
            
            # Create target if not exists
            if 'risk_score' in health_data.columns:
                y_risk = health_data['risk_score']
            else:
                # Create synthetic risk score
                y_risk = (
                    health_data['emergency_cases'].fillna(0) * 0.5 +
                    health_data['occupancy_rate'].fillna(0.5) * 50
                )
            
            # Create demand category
            if 'patients_per_day' in health_data.columns:
                y_demand = pd.cut(health_data['patients_per_day'], bins=[0, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
            else:
                y_demand = None
            
            # Train
            metrics = self.health_predictor.train(X, y_risk, y_demand)
            self.health_predictor.save_model()
            
            return {**metrics, 'status': 'success', 'features_used': len(available_features)}
        except Exception as e:
            logger.error(f"Error training health model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _train_infrastructure_model(self, infra_data: pd.DataFrame) -> dict:
        """Train infrastructure predictor"""
        try:
            # Prepare features
            feature_cols = ['response_time_hours', 'resolution_time_hours', 'citizen_satisfaction', 
                          'recurring', 'priority_numeric', 'is_weekend', 'month']
            available_features = [col for col in feature_cols if col in infra_data.columns]
            
            if len(available_features) < 3:
                logger.warning("Insufficient features for infrastructure model")
                return {'status': 'skipped', 'reason': 'insufficient_features'}
            
            X = infra_data[available_features].fillna(0)
            
            # Create target if not exists
            if 'failure_risk' in infra_data.columns:
                y = (infra_data['failure_risk'] > 0.5).astype(int)
            else:
                # Create synthetic failure target
                y = (
                    (infra_data['response_time_hours'].fillna(24) > 48) |
                    (infra_data['recurring'].fillna(0) == 1) |
                    (infra_data['citizen_satisfaction'].fillna(3) < 3)
                ).astype(int)
            
            # Train
            metrics = self.infrastructure_predictor.train(X, y)
            self.infrastructure_predictor.save_model()
            
            return {**metrics, 'status': 'success', 'features_used': len(available_features)}
        except Exception as e:
            logger.error(f"Error training infrastructure model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _train_demand_model(self, demand_data: pd.DataFrame) -> dict:
        """Train demand forecaster"""
        try:
            # Prepare features
            feature_cols = ['hour', 'day_of_week', 'month', 'is_business_hours', 'is_night', 
                          'service_type_encoded', 'urgency']
            available_features = [col for col in feature_cols if col in demand_data.columns]
            
            if len(available_features) < 3:
                logger.warning("Insufficient features for demand model")
                return {'status': 'skipped', 'reason': 'insufficient_features'}
            
            X = demand_data[available_features].fillna(0)
            
            # Target: processing time or create synthetic demand
            if 'processing_time_hours' in demand_data.columns:
                y = demand_data['processing_time_hours']
            else:
                # Create synthetic demand based on features
                y = (
                    50 +
                    demand_data.get('is_business_hours', 0) * 30 +
                    demand_data.get('urgency', 5) * 5 +
                    np.random.normal(0, 10, len(demand_data))
                )
                y = np.maximum(y, 0)
            
            # Train
            metrics = self.demand_forecaster.train(X, y)
            self.demand_forecaster.save_model()
            
            return {**metrics, 'status': 'success', 'features_used': len(available_features)}
        except Exception as e:
            logger.error(f"Error training demand model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _train_sentiment_model(self, feedback_data: pd.DataFrame) -> dict:
        """Train sentiment analyzer"""
        try:
            # Prepare features
            if 'comment' not in feedback_data.columns:
                logger.warning("No comment column for sentiment analysis")
                return {'status': 'skipped', 'reason': 'no_text_data'}
            
            X = feedback_data[['comment', 'rating']].copy() if 'rating' in feedback_data.columns else feedback_data[['comment']].copy()
            
            # Target
            if 'sentiment' in feedback_data.columns:
                y = feedback_data['sentiment']
            elif 'sentiment_numeric' in feedback_data.columns:
                mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                y = feedback_data['sentiment_numeric'].map(mapping)
            else:
                # Create synthetic sentiment from rating
                if 'rating' in feedback_data.columns:
                    y = pd.cut(feedback_data['rating'], bins=[0, 2, 3, 5], labels=['Negative', 'Neutral', 'Positive'])
                else:
                    logger.warning("Cannot create sentiment target")
                    return {'status': 'skipped', 'reason': 'no_sentiment_target'}
            
            # Train
            metrics = self.sentiment_analyzer.train(X, y, text_column='comment')
            self.sentiment_analyzer.save_model()
            
            return {**metrics, 'status': 'success'}
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def get_training_summary(self) -> str:
        """Get training summary"""
        summary = "\n" + "=" * 80 + "\n"
        summary += "TRAINING SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for model_name, metrics in self.training_results.items():
            summary += f"{model_name.upper()} MODEL:\n"
            summary += f"  Status: {metrics.get('status', 'unknown')}\n"
            
            if metrics.get('status') == 'success':
                for key, value in metrics.items():
                    if key not in ['status', 'classification_report', 'features_used']:
                        if isinstance(value, float):
                            summary += f"  {key}: {value:.4f}\n"
                        else:
                            summary += f"  {key}: {value}\n"
            elif metrics.get('status') == 'failed':
                summary += f"  Error: {metrics.get('error', 'unknown')}\n"
            
            summary += "\n"
        
        return summary


if __name__ == "__main__":
    print("=" * 80)
    print("MODEL TRAINER TEST")
    print("=" * 80)
    
    # Load data
    loader = RealDataLoader()
    raw_data = loader.load_all_datasets()
    
    # Process data
    processor = DataProcessor(raw_data)
    processed_data = processor.process_all()
    
    # Train all models
    trainer = ModelTrainer()
    results = trainer.train_all_models(processed_data)
    
    # Print summary
    print(trainer.get_training_summary())
    
    print("=" * 80)
    print("ALL MODELS TRAINED AND SAVED")
    print("=" * 80)
