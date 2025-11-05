"""
Quick script to train all models immediately
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from data.data_loader import RealDataLoader
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer

print("=" * 80)
print("TRAINING ALL MODELS")
print("=" * 80)

try:
    # Load data
    print("\n[1/3] Loading datasets...")
    loader = RealDataLoader()
    raw_data = loader.load_all_datasets()
    print(f">> Loaded {len(raw_data)} datasets")
    
    # Process data
    print("\n[2/3] Processing data...")
    processor = DataProcessor(raw_data)
    processed_data = processor.process_all()
    print(f">> Processed {len(processed_data)} datasets")
    
    # Train models
    print("\n[3/3] Training models...")
    trainer = ModelTrainer()
    results = trainer.train_all_models(processed_data)
    
    print("\n" + "=" * 80)
    print(">> ALL MODELS TRAINED AND SAVED!")
    print("=" * 80)
    print("\nModels saved to: models/saved_models/")
    print("Now restart the Flask app: python app.py")
    print("=" * 80)
    
except Exception as e:
    print(f"\n!! Error: {e}")
    import traceback
    traceback.print_exc()
