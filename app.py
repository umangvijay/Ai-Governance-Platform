"""
Flask Web Application - AI Governance Platform Frontend
Run with: python app.py
Access at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import RealDataLoader
from data.data_processor import DataProcessor
from models.health_predictor import HealthRiskPredictor
from models.infrastructure_predictor import InfrastructureFailurePredictor
from models.demand_forecaster import DemandForecaster
from models.sentiment_analyzer import SentimentAnalyzer
from models.whatif_analyzer import HealthWhatIfAnalyzer, InfrastructureWhatIfAnalyzer, DemandWhatIfAnalyzer
from rag.gemini_rag import GeminiRAG
from privacy.anonymizer import DataAnonymizer
from utils.dataset_downloader import DatasetDownloader
from data.multisectoral_merger import MultiSectoralMerger

import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for models
platform_state = {
    'initialized': False,
    'data_loader': None,
    'dataset_downloader': None,
    'multi_sectoral_merger': None,
    'models': {},
    'whatif_analyzers': {},
    'rag_system': None,
    'anonymizer': None,
    'stats': {},
    'merged_data': None
}


def initialize_platform():
    """Initialize platform on first request"""
    if platform_state['initialized']:
        return
    
    try:
        logger.info("Initializing platform...")
        
        # Load data
        platform_state['data_loader'] = RealDataLoader()
        raw_data = platform_state['data_loader'].load_all_datasets()
        
        # Process data
        processor = DataProcessor(raw_data)
        processed_data = processor.process_all()
        
        # Initialize models
        platform_state['models']['health'] = HealthRiskPredictor()
        platform_state['models']['infrastructure'] = InfrastructureFailurePredictor()
        platform_state['models']['demand'] = DemandForecaster()
        platform_state['models']['sentiment'] = SentimentAnalyzer()
        
        # Try to load saved models, otherwise train new ones
        model_dir = Path('models/saved_models')
        models_loaded = False
        
        if model_dir.exists():
            try:
                logger.info("Attempting to load saved models...")
                platform_state['models']['health'].load_model()
                platform_state['models']['infrastructure'].load_model()
                platform_state['models']['demand'].load_model()
                platform_state['models']['sentiment'].load_model()
                logger.info("✓ Loaded saved models successfully")
                models_loaded = True
            except Exception as e:
                logger.warning(f"Could not load saved models: {e}")
                models_loaded = False
        
        # Train models if not loaded
        if not models_loaded:
            logger.info("Training new models...")
            try:
                from models.model_trainer import ModelTrainer
                trainer = ModelTrainer()
                results = trainer.train_all_models(processed_data)
                
                # Update platform state with trained models
                platform_state['models'] = {
                    'health': trainer.health_predictor,
                    'infrastructure': trainer.infrastructure_predictor,
                    'demand': trainer.demand_forecaster,
                    'sentiment': trainer.sentiment_analyzer
                }
                
                logger.info("✓ All models trained successfully")
                logger.info(trainer.get_training_summary())
            except Exception as e:
                logger.error(f"Error training models: {e}")
                import traceback
                traceback.print_exc()
                # Keep the initialized models even if training fails
                # They will return errors when predict is called
        
        # Initialize RAG with actual data
        try:
            platform_state['rag_system'] = GeminiRAG()
            
            # Create detailed documents from actual data
            rag_docs = []
            
            for dataset_name, df in raw_data.items():
                # Add dataset summary
                rag_docs.append({
                    'text': f"Dataset: {dataset_name}\nRecords: {len(df)}\nColumns: {', '.join(df.columns.tolist())}",
                    'metadata': {'source': dataset_name, 'type': 'summary'}
                })
                
                # For hospital data, index actual hospital details
                if 'hospital' in dataset_name.lower() or 'health' in dataset_name.lower():
                    # Index each hospital as a document
                    for idx, row in df.head(100).iterrows():  # Index first 100 records
                        hospital_info = []
                        for col in df.columns:
                            if pd.notna(row[col]):
                                hospital_info.append(f"{col}: {row[col]}")
                        
                        if hospital_info:
                            rag_docs.append({
                                'text': f"Hospital/Health Facility:\n" + "\n".join(hospital_info[:10]),  # First 10 fields
                                'metadata': {'source': dataset_name, 'type': 'hospital', 'index': idx}
                            })
                
                # For other datasets, create sample documents
                elif len(df) > 0 and len(df) <= 50:
                    # Small datasets - index all records
                    for idx, row in df.iterrows():
                        row_text = []
                        for col in df.columns:
                            if pd.notna(row[col]):
                                row_text.append(f"{col}: {row[col]}")
                        if row_text:
                            rag_docs.append({
                                'text': "\n".join(row_text[:8]),
                                'metadata': {'source': dataset_name, 'type': 'record', 'index': idx}
                            })
                elif len(df) > 0:
                    # Large datasets - index sample records
                    sample_df = df.sample(min(30, len(df)))
                    for idx, row in sample_df.iterrows():
                        row_text = []
                        for col in df.columns:
                            if pd.notna(row[col]):
                                row_text.append(f"{col}: {row[col]}")
                        if row_text:
                            rag_docs.append({
                                'text': "\n".join(row_text[:8]),
                                'metadata': {'source': dataset_name, 'type': 'sample', 'index': idx}
                            })
            
            logger.info(f"Indexing {len(rag_docs)} documents for RAG...")
            platform_state['rag_system'].index_documents(rag_docs)
            logger.info(f"✓ RAG system ready with {len(rag_docs)} documents")
            
        except Exception as e:
            logger.warning(f"RAG system error: {e}")
            platform_state['rag_system'] = None
        
        # Initialize anonymizer
        platform_state['anonymizer'] = DataAnonymizer()
        
        # Initialize dataset downloader
        platform_state['dataset_downloader'] = DatasetDownloader()
        
        # Initialize What-If Analyzers
        platform_state['whatif_analyzers']['health'] = HealthWhatIfAnalyzer()
        platform_state['whatif_analyzers']['infrastructure'] = InfrastructureWhatIfAnalyzer()
        platform_state['whatif_analyzers']['demand'] = DemandWhatIfAnalyzer()
        
        # Set baseline data for what-if analyzers
        platform_state['whatif_analyzers']['health'].set_baseline({
            'beds': 1000,
            'doctors': 100,
            'nurses': 200,
            'patients': 800,
            'base_demand': 100,
            'growth_rate': 0.02
        })
        
        # Store stats
        platform_state['stats'] = platform_state['data_loader'].get_statistics()
        
        platform_state['initialized'] = True
        logger.info("Platform initialized successfully")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        import traceback
        traceback.print_exc()


@app.route('/')
def index():
    """Home page"""
    initialize_platform()
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Get platform statistics"""
    initialize_platform()
    
    stats = platform_state.get('stats', {})
    models = platform_state.get('models', {})
    
    return jsonify({
        'total_datasets': stats.get('total_datasets', 0),
        'total_records': stats.get('total_records', 0),
        'models_loaded': len([m for m in models.values() if m and hasattr(m, 'is_trained') and m.is_trained]),
        'rag_available': platform_state.get('rag_system') is not None
    })


@app.route('/api/test/gemini', methods=['GET'])
def test_gemini():
    """Test if Gemini API is working"""
    try:
        import os
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'GEMINI_API_KEY not found in .env file',
                'suggestion': 'Add GEMINI_API_KEY=your_key to .env file'
            })
        
        # Try to initialize RAG
        from rag.gemini_rag import GeminiRAG
        rag = GeminiRAG(api_key=api_key)
        
        return jsonify({
            'status': 'success',
            'message': 'Gemini API is working!',
            'api_key_present': True,
            'api_key_length': len(api_key),
            'model': 'gemini-pro-latest'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'suggestion': 'Check if your API key is valid'
        })


@app.route('/api/predict/health', methods=['POST'])
def predict_health():
    """Predict health risk"""
    initialize_platform()
    
    try:
        data = request.json
        days = data.get('days', 7)
        recent_values = data.get('recent_values', [50, 52, 48, 55, 53, 51, 54])
        
        model = platform_state.get('models', {}).get('health')
        
        if not model:
            return jsonify({
                'error': 'Health model not initialized',
                'forecast': [0] * days,
                'days': days,
                'average': 0,
                'trend': 'unknown'
            })
        
        if not hasattr(model, 'is_trained') or not model.is_trained:
            return jsonify({
                'error': 'Model not trained yet',
                'forecast': [0] * days,
                'days': days,
                'average': 0,
                'trend': 'unknown'
            })
        
        forecast = model.forecast_next_days(recent_values, days=days)
        
        if not forecast or len(forecast) == 0:
            forecast = [0] * days
        
        return jsonify({
            'forecast': [float(f) for f in forecast],
            'days': days,
            'average': float(np.mean(forecast)) if forecast else 0,
            'trend': 'increasing' if forecast and forecast[-1] > forecast[0] else 'decreasing'
        })
        
    except Exception as e:
        logger.error(f"Health prediction error: {e}")
        return jsonify({
            'error': str(e),
            'forecast': [0] * data.get('days', 7),
            'days': data.get('days', 7),
            'average': 0,
            'trend': 'unknown'
        })


@app.route('/api/predict/infrastructure', methods=['POST'])
def predict_infrastructure():
    """Predict infrastructure failure risk"""
    initialize_platform()
    
    try:
        data = request.json
        
        # Create sample features
        features = pd.DataFrame({
            'response_time_hours': [data.get('response_time', 24)],
            'resolution_time_hours': [data.get('resolution_time', 48)],
            'citizen_satisfaction': [data.get('satisfaction', 3)],
            'recurring': [data.get('recurring', 0)],
            'priority_numeric': [data.get('priority', 1)],
            'is_weekend': [data.get('is_weekend', 0)],
            'month': [data.get('month', 1)]
        })
        
        model = platform_state['models']['infrastructure']
        if not model or not model.is_trained:
            return jsonify({'error': 'Model not trained'}), 400
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        failure_prob = float(probability[1] if len(probability) > 1 else probability[0])
        
        return jsonify({
            'failure_prediction': int(prediction),
            'failure_probability': failure_prob,
            'risk_level': 'HIGH' if failure_prob > 0.7 else ('MEDIUM' if failure_prob > 0.4 else 'LOW'),
            'recommendation': 'Immediate action required' if failure_prob > 0.7 else 'Monitor closely' if failure_prob > 0.4 else 'Normal maintenance'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/demand', methods=['POST'])
def predict_demand():
    """Forecast service demand"""
    initialize_platform()
    
    try:
        data = request.json
        hours = data.get('hours', 24)
        
        # Create recent data
        recent_data = pd.DataFrame({
            'hour': [12],
            'day_of_week': [0],
            'month': [1],
            'is_business_hours': [1],
            'is_night': [0],
            'service_type_encoded': [0],
            'urgency': [5]
        })
        
        model = platform_state['models']['demand']
        if not model or not model.is_trained:
            return jsonify({'error': 'Model not trained'}), 400
        
        forecast = model.forecast_next_hours(recent_data, hours=hours)
        
        return jsonify({
            'forecast': forecast[:24],  # Limit to 24 hours for display
            'hours': len(forecast),
            'average_demand': np.mean([f['predicted_demand'] for f in forecast]),
            'peak_hour': max(forecast, key=lambda x: x['predicted_demand'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of text"""
    initialize_platform()
    
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        model = platform_state.get('models', {}).get('sentiment')
        
        if not model:
            return jsonify({
                'error': 'Sentiment analyzer not initialized',
                'sentiment': 'Unknown',
                'confidence': 0,
                'probabilities': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            }), 400
        
        if not hasattr(model, 'is_trained') or not model.is_trained:
            return jsonify({
                'error': 'Sentiment model not trained yet. Please wait for model training to complete.',
                'sentiment': 'Unknown',
                'confidence': 0,
                'probabilities': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
            }), 400
        
        # Create DataFrame with only text (model was trained without rating)
        df = pd.DataFrame({
            'comment': [text]
        })
        
        sentiment = model.predict(df, text_column='comment')[0]
        probabilities = model.predict_proba(df, text_column='comment')[0]
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(max(probabilities)),
            'probabilities': {
                'Positive': float(probabilities[2]) if len(probabilities) > 2 else 0,
                'Neutral': float(probabilities[1]) if len(probabilities) > 1 else 0,
                'Negative': float(probabilities[0]) if len(probabilities) > 0 else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        return jsonify({
            'error': str(e),
            'sentiment': 'Unknown',
            'confidence': 0,
            'probabilities': {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        }), 500


@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    """RAG query with Gemini"""
    initialize_platform()
    
    try:
        data = request.json
        question = data.get('question', '')
        
        if not platform_state['rag_system']:
            return jsonify({'error': 'RAG system not available'}), 400
        
        # Check if this is a hospital-related query
        is_hospital_query = any(word in question.lower() for word in ['hospital', 'health', 'clinic', 'medical'])
        
        # Retrieve more documents for hospital queries
        top_k = 10 if is_hospital_query else 5
        
        result = platform_state['rag_system'].query(question, top_k=top_k)
        
        # Enhance answer for hospital queries
        if is_hospital_query and result['sources']:
            hospital_sources = [s for s in result['sources'] if s['metadata'].get('type') == 'hospital']
            
            if hospital_sources:
                # Extract hospital names and details
                hospital_info = []
                for source in hospital_sources[:10]:  # Top 10 hospitals
                    text = source['text']
                    hospital_info.append(text)
                
                # Add structured hospital list to answer
                enhanced_answer = result['answer']
                if hospital_info:
                    enhanced_answer += "\n\n📋 Hospital Details Found:\n"
                    for i, info in enumerate(hospital_info[:5], 1):
                        enhanced_answer += f"\n{i}. {info[:200]}..."  # First 200 chars
                
                result['answer'] = enhanced_answer
        
        return jsonify({
            'question': question,
            'answer': result['answer'],
            'sources': result['sources'][:3],  # Return top 3 sources to UI
            'n_sources': result['n_sources']
        })
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/anonymize', methods=['POST'])
def anonymize_text():
    """Anonymize PII in text"""
    initialize_platform()
    
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        anonymized = platform_state['anonymizer'].mask_text(text)
        pii_detected = platform_state['anonymizer'].detect_pii(text)
        
        return jsonify({
            'original': text,
            'anonymized': anonymized,
            'pii_count': len(pii_detected),
            'pii_types': [p['type'] for p in pii_detected]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/whatif/health', methods=['POST'])
def whatif_health():
    """Health what-if scenario analysis"""
    initialize_platform()
    
    try:
        data = request.json
        scenario_type = data.get('scenario_type', 'bed_increase')
        days = data.get('days', 30)
        
        analyzer = platform_state['whatif_analyzers']['health']
        
        if scenario_type == 'bed_increase':
            percent = data.get('percent', 20)
            scenario = analyzer.analyze_bed_increase(percent, days)
        elif scenario_type == 'staff_increase':
            doctors_percent = data.get('doctors_percent', 15)
            nurses_percent = data.get('nurses_percent', 25)
            scenario = analyzer.analyze_staff_increase(doctors_percent, nurses_percent, days)
        elif scenario_type == 'outbreak':
            severity = data.get('severity', 'moderate')
            scenario = analyzer.analyze_outbreak_response(severity, days)
        else:
            return jsonify({'error': 'Invalid scenario type'}), 400
        
        return jsonify({
            'scenario': scenario['name'],
            'forecast_days': scenario['forecast_days'],
            'forecast': [float(f) for f in scenario['results']['forecast']],
            'impact': {
                'average_change': float(scenario['results']['impact']['average_change']),
                'peak_change': float(scenario['results']['impact']['peak_change']),
                'percent_change': float(scenario['results']['impact']['percent_change'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/whatif/infrastructure', methods=['POST'])
def whatif_infrastructure():
    """Infrastructure what-if scenario analysis"""
    initialize_platform()
    
    try:
        data = request.json
        scenario_type = data.get('scenario_type', 'response_time')
        days = data.get('days', 30)
        percent = data.get('percent', 20)
        
        analyzer = platform_state['whatif_analyzers']['infrastructure']
        
        if scenario_type == 'response_time':
            scenario = analyzer.analyze_response_time_improvement(percent, days)
        elif scenario_type == 'maintenance':
            scenario = analyzer.analyze_maintenance_increase(percent, days)
        else:
            return jsonify({'error': 'Invalid scenario type'}), 400
        
        return jsonify({
            'scenario': scenario['name'],
            'forecast_days': scenario['forecast_days'],
            'forecast': [float(f) for f in scenario['results']['forecast']],
            'impact': {
                'average_change': float(scenario['results']['impact']['average_change']),
                'peak_change': float(scenario['results']['impact']['peak_change']),
                'percent_change': float(scenario['results']['impact']['percent_change'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/whatif/demand', methods=['POST'])
def whatif_demand():
    """Demand what-if scenario analysis"""
    initialize_platform()
    
    try:
        data = request.json
        scenario_type = data.get('scenario_type', 'service_expansion')
        days = data.get('days', 30)
        
        analyzer = platform_state['whatif_analyzers']['demand']
        
        if scenario_type == 'service_expansion':
            service_type = data.get('service_type', 'healthcare')
            percent = data.get('percent', 25)
            scenario = analyzer.analyze_service_expansion(service_type, percent, days)
        elif scenario_type == 'seasonal_surge':
            surge_factor = data.get('surge_factor', 1.5)
            scenario = analyzer.analyze_seasonal_surge(surge_factor, days)
        else:
            return jsonify({'error': 'Invalid scenario type'}), 400
        
        return jsonify({
            'scenario': scenario['name'],
            'forecast_days': scenario['forecast_days'],
            'forecast': [float(f) for f in scenario['results']['forecast']],
            'impact': {
                'average_change': float(scenario['results']['impact']['average_change']),
                'peak_change': float(scenario['results']['impact']['peak_change']),
                'percent_change': float(scenario['results']['impact']['percent_change'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/datasets', methods=['POST'])
def download_datasets():
    """Download datasets from data.gov.in"""
    initialize_platform()
    
    try:
        downloader = platform_state['dataset_downloader']
        datasets = downloader.download_all()
        stats = downloader.get_statistics(datasets)
        
        return jsonify({
            'total_datasets': stats['total_datasets'],
            'total_records': stats['total_records'],
            'datasets': {k: v for k, v in stats['datasets'].items()}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("AI GOVERNANCE PLATFORM - WEB INTERFACE")
    print("=" * 80)
    print("\n>> Starting web server...")
    print("\n>> Access the platform at: http://localhost:5000")
    print("\n>> Press CTRL+C to stop the server\n")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
