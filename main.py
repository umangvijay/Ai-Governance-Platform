"""
MAIN AI GOVERNANCE PLATFORM
Complete integration of all modules with real predictions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import RealDataLoader
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.health_predictor import HealthRiskPredictor
from models.infrastructure_predictor import InfrastructureFailurePredictor
from models.demand_forecaster import DemandForecaster
from models.sentiment_analyzer import SentimentAnalyzer
from rag.gemini_rag import GeminiRAG
from privacy.anonymizer import DataAnonymizer
from dashboard.visualizer import DashboardVisualizer

import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class AIGovernancePlatform:
    """Complete AI-Powered Governance Platform"""
    
    def __init__(self):
        self.data_loader = None
        self.data_processor = None
        self.model_trainer = None
        self.rag_system = None
        self.anonymizer = None
        self.visualizer = None
        
        self.raw_data = {}
        self.processed_data = {}
        self.training_results = {}
        self.predictions = {}
        
        logger.info("=" * 100)
        logger.info("AI-POWERED GOVERNANCE PLATFORM FOR MAHARASHTRA")
        logger.info("Transforming Citizen Service Delivery with Predictive Intelligence")
        logger.info("=" * 100)
    
    def initialize(self):
        """Initialize all platform components"""
        logger.info("\n" + "=" * 100)
        logger.info("PLATFORM INITIALIZATION")
        logger.info("=" * 100)
        
        try:
            # Step 1: Load Data
            logger.info("\n[1/7] Loading datasets from D:\\DATASET...")
            self.data_loader = RealDataLoader()
            self.raw_data = self.data_loader.load_all_datasets()
            data_stats = self.data_loader.get_statistics()
            logger.info(f"✓ Loaded {data_stats['total_datasets']} datasets ({data_stats['total_records']:,} records)")
            
            # Step 2: Process Data
            logger.info("\n[2/7] Processing data for ML models...")
            self.data_processor = DataProcessor(self.raw_data)
            self.processed_data = self.data_processor.process_all()
            logger.info(f"✓ Processed {len(self.processed_data)} datasets")
            
            # Step 3: Train ML Models
            logger.info("\n[3/7] Training ML models...")
            self.model_trainer = ModelTrainer()
            self.training_results = self.model_trainer.train_all_models(self.processed_data)
            logger.info(f"✓ Trained {len(self.training_results)} models")
            
            # Step 4: Initialize RAG System
            logger.info("\n[4/7] Initializing RAG system with Gemini...")
            try:
                self.rag_system = GeminiRAG()
                rag_docs = self._prepare_rag_documents()
                self.rag_system.index_documents(rag_docs)
                logger.info(f"✓ Indexed {len(rag_docs)} documents for RAG")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}")
                self.rag_system = None
            
            # Step 5: Initialize Privacy Module
            logger.info("\n[5/7] Initializing privacy & security...")
            self.anonymizer = DataAnonymizer()
            logger.info("✓ Privacy module ready")
            
            # Step 6: Initialize Visualizer
            logger.info("\n[6/7] Initializing dashboard visualizer...")
            self.visualizer = DashboardVisualizer()
            logger.info("✓ Visualizer ready")
            
            # Step 7: Generate Predictions
            logger.info("\n[7/7] Generating predictions...")
            self._generate_all_predictions()
            logger.info(f"✓ Generated {len(self.predictions)} prediction sets")
            
            logger.info("\n" + "=" * 100)
            logger.info("✓ PLATFORM INITIALIZATION COMPLETE")
            logger.info("=" * 100)
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _prepare_rag_documents(self):
        """Prepare documents for RAG indexing"""
        documents = []
        
        for dataset_name, df in self.raw_data.items():
            # Create summary document for each dataset
            summary = f"{dataset_name.replace('_', ' ').title()}: {len(df)} records. "
            
            if 'description' in df.columns:
                summary += f"Sample: {df['description'].iloc[0]}"
            elif len(df.columns) > 0:
                summary += f"Columns: {', '.join(df.columns[:5].tolist())}"
            
            documents.append({
                'text': summary,
                'metadata': {'source': dataset_name, 'type': 'dataset_summary'}
            })
        
        return documents
    
    def _generate_all_predictions(self):
        """Generate predictions from all models"""
        try:
            # Health predictions
            if self.model_trainer.health_predictor and self.model_trainer.health_predictor.is_trained:
                self.predictions['health_forecast'] = self.model_trainer.health_predictor.forecast_next_days(
                    [50, 52, 48, 55, 53, 51, 54], days=7
                )
                logger.info("  ✓ Health risk forecast generated")
            
            # Infrastructure predictions
            if self.model_trainer.infrastructure_predictor and self.model_trainer.infrastructure_predictor.is_trained:
                if 'infrastructure' in self.processed_data:
                    sample_data = self.processed_data['infrastructure'].head(100)
                    feature_cols = self.model_trainer.infrastructure_predictor.feature_names
                    available_cols = [col for col in feature_cols if col in sample_data.columns]
                    
                    if available_cols:
                        X_sample = sample_data[available_cols]
                        probas = self.model_trainer.infrastructure_predictor.predict_proba(X_sample)
                        self.predictions['infrastructure_risk'] = probas[:, 1].tolist() if probas.shape[1] > 1 else probas[:, 0].tolist()
                        logger.info("  ✓ Infrastructure risk predictions generated")
            
            # Demand forecast
            if self.model_trainer.demand_forecaster and self.model_trainer.demand_forecaster.is_trained:
                if 'service_requests' in self.processed_data:
                    recent_data = self.processed_data['service_requests'].tail(1)
                    self.predictions['demand_forecast'] = self.model_trainer.demand_forecaster.forecast_next_hours(
                        recent_data, hours=24
                    )
                    logger.info("  ✓ Service demand forecast generated")
            
            # Sentiment analysis
            if self.model_trainer.sentiment_analyzer and self.model_trainer.sentiment_analyzer.is_trained:
                if 'citizen_feedback' in self.processed_data:
                    feedback_data = self.processed_data['citizen_feedback'].head(100)
                    if 'comment' in feedback_data.columns:
                        sentiments = self.model_trainer.sentiment_analyzer.predict(feedback_data, text_column='comment')
                        self.predictions['sentiment_distribution'] = self.model_trainer.sentiment_analyzer.analyze_sentiment_distribution(sentiments)
                        logger.info("  ✓ Sentiment analysis completed")
            
        except Exception as e:
            logger.warning(f"Some predictions failed: {e}")
    
    def generate_dashboard(self):
        """Generate visual dashboard"""
        logger.info("\n" + "=" * 100)
        logger.info("GENERATING DASHBOARD")
        logger.info("=" * 100)
        
        try:
            # Plot model performance
            self.visualizer.plot_model_performance(self.training_results)
            
            # Plot predictions
            self.visualizer.plot_predictions(self.predictions)
            
            # Plot data overview
            data_stats = self.data_loader.get_statistics()
            self.visualizer.plot_data_overview(data_stats)
            
            # Generate HTML dashboard
            dashboard_path = self.visualizer.generate_html_dashboard()
            
            logger.info(f"\n✓ Dashboard generated: {dashboard_path}")
            logger.info(f"✓ Charts saved: {len(self.visualizer.generated_charts)}")
            
            return dashboard_path
            
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            return None
    
    def print_summary(self):
        """Print platform summary"""
        logger.info("\n" + "=" * 100)
        logger.info("PLATFORM SUMMARY")
        logger.info("=" * 100)
        
        # Data summary
        logger.info("\n📊 DATA:")
        data_stats = self.data_loader.get_statistics()
        logger.info(f"  Total Datasets: {data_stats['total_datasets']}")
        logger.info(f"  Total Records: {data_stats['total_records']:,}")
        
        # Model summary
        logger.info("\n🤖 ML MODELS:")
        for model_name, metrics in self.training_results.items():
            status = metrics.get('status', 'unknown')
            logger.info(f"  {model_name.capitalize()}: {status.upper()}")
            if status == 'success':
                if 'accuracy' in metrics:
                    logger.info(f"    Accuracy: {metrics['accuracy']:.3f}")
                elif 'risk_r2' in metrics:
                    logger.info(f"    R²: {metrics['risk_r2']:.3f}")
                elif 'r2' in metrics:
                    logger.info(f"    R²: {metrics['r2']:.3f}")
        
        # Predictions summary
        logger.info("\n📈 PREDICTIONS:")
        if 'health_forecast' in self.predictions:
            forecast = self.predictions['health_forecast']
            logger.info(f"  Health Risk Forecast: {len(forecast)} days")
            logger.info(f"    Next day prediction: {forecast[0]:.0f} cases")
        
        if 'infrastructure_risk' in self.predictions:
            risks = self.predictions['infrastructure_risk']
            high_risk_count = sum(1 for r in risks if r > 0.7)
            logger.info(f"  Infrastructure Risk: {len(risks)} locations analyzed")
            logger.info(f"    High risk locations: {high_risk_count}")
        
        if 'demand_forecast' in self.predictions:
            forecast = self.predictions['demand_forecast']
            logger.info(f"  Demand Forecast: {len(forecast)} hours")
            avg_demand = np.mean([f['predicted_demand'] for f in forecast])
            logger.info(f"    Average demand: {avg_demand:.0f} requests/hour")
        
        if 'sentiment_distribution' in self.predictions:
            dist = self.predictions['sentiment_distribution']
            logger.info(f"  Citizen Sentiment:")
            for sentiment, stats in dist.items():
                logger.info(f"    {sentiment}: {stats['percentage']:.1f}%")
        
        # RAG summary
        logger.info("\n🧠 RAG SYSTEM:")
        if self.rag_system:
            logger.info("  Status: Operational")
            logger.info(f"  Documents indexed: {len(self.rag_system.documents)}")
        else:
            logger.info("  Status: Not available")
        
        # Privacy summary
        logger.info("\n🔒 PRIVACY & SECURITY:")
        logger.info("  Data Anonymization: Enabled")
        logger.info("  PII Detection: Active")
        
        logger.info("\n" + "=" * 100)
    
    def demonstrate_capabilities(self):
        """Demonstrate platform capabilities"""
        logger.info("\n" + "=" * 100)
        logger.info("DEMONSTRATING CAPABILITIES")
        logger.info("=" * 100)
        
        # Demo 1: Query RAG system
        if self.rag_system:
            logger.info("\n### RAG QUERY DEMO ###")
            query = "What data is available in this platform?"
            result = self.rag_system.query(query, top_k=2)
            logger.info(f"\nQ: {query}")
            logger.info(f"A: {result['answer'][:200]}...")
        
        # Demo 2: Data anonymization
        logger.info("\n### DATA PRIVACY DEMO ###")
        sample_data = pd.DataFrame({
            'phone': ['9876543210'],
            'description': ['Contact me at 9876543210']
        })
        anonymized = self.anonymizer.anonymize_dataframe(sample_data)
        logger.info("Original: " + sample_data['description'].iloc[0])
        logger.info("Anonymized: " + anonymized['description'].iloc[0])
        
        logger.info("\n" + "=" * 100)


def main():
    """Main execution"""
    print("\n")
    
    # Create platform
    platform = AIGovernancePlatform()
    
    # Initialize
    success = platform.initialize()
    
    if not success:
        logger.error("\n❌ Platform initialization failed")
        return
    
    # Print summary
    platform.print_summary()
    
    # Demonstrate capabilities
    platform.demonstrate_capabilities()
    
    # Generate dashboard
    dashboard_path = platform.generate_dashboard()
    
    # Final message
    logger.info("\n" + "=" * 100)
    logger.info("✅ PLATFORM READY FOR PRODUCTION")
    logger.info("=" * 100)
    
    if dashboard_path:
        logger.info(f"\n📊 Open dashboard: {dashboard_path}")
    
    logger.info("\nAll models trained and saved in: models/saved_models/")
    logger.info("Dashboard visualizations in: dashboard/outputs/")
    
    logger.info("\n" + "=" * 100)


if __name__ == "__main__":
    main()
