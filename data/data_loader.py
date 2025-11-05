"""
Real Data Loader from D:\DATASET with synthetic augmentation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RealDataLoader:
    """Load real data from D:\DATASET and augment with synthetic data"""
    
    def __init__(self):
        self.dataset_path = os.getenv('DATASET_PATH', r'D:\DATASET')
        self.dataset_new_path = os.getenv('DATASET_NEW_PATH', r'D:\DATASET WITH NEW DATA Set')
        self.data = {}
        
        logger.info(f"DataLoader initialized")
        logger.info(f"Primary path: {self.dataset_path}")
        logger.info(f"Secondary path: {self.dataset_new_path}")
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets"""
        logger.info("=" * 80)
        logger.info("LOADING REAL DATASETS")
        logger.info("=" * 80)
        
        # Load from primary path
        self._load_from_directory(self.dataset_path)
        
        # Load from secondary path if exists
        if os.path.exists(self.dataset_new_path):
            self._load_from_directory(self.dataset_new_path)
        
        # Augment with synthetic data if needed
        self._augment_datasets()
        
        logger.info(f"\n✓ Total datasets loaded: {len(self.data)}")
        for name, df in self.data.items():
            logger.info(f"  - {name}: {len(df)} records, {len(df.columns)} columns")
        
        return self.data
    
    def _load_from_directory(self, directory: str):
        """Load all CSV and Excel files from directory"""
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return
        
        path = Path(directory)
        
        # Load CSV files
        for csv_file in path.glob('*.csv'):
            try:
                logger.info(f"Loading: {csv_file.name}")
                df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
                dataset_name = csv_file.stem.lower().replace(' ', '_')
                
                if dataset_name in self.data:
                    # Merge if already exists
                    self.data[dataset_name] = pd.concat([self.data[dataset_name], df], ignore_index=True)
                else:
                    self.data[dataset_name] = df
                
                logger.info(f"✓ Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error loading {csv_file.name}: {e}")
        
        # Load Excel files
        for excel_file in path.glob('*.xls*'):
            try:
                logger.info(f"Loading: {excel_file.name}")
                df = pd.read_excel(excel_file)
                dataset_name = excel_file.stem.lower().replace(' ', '_').replace('-', '_')
                
                if dataset_name in self.data:
                    self.data[dataset_name] = pd.concat([self.data[dataset_name], df], ignore_index=True)
                else:
                    self.data[dataset_name] = df
                
                logger.info(f"✓ Loaded {len(df)} records from {excel_file.name}")
            except Exception as e:
                logger.error(f"Error loading {excel_file.name}: {e}")
    
    def _augment_datasets(self):
        """Augment with synthetic data for better model training"""
        logger.info("\n--- Augmenting datasets with synthetic data ---")
        
        # Generate synthetic health data if needed
        if not any('health' in key or 'hospital' in key for key in self.data.keys()):
            logger.info("Generating synthetic health infrastructure data...")
            self.data['health_infrastructure'] = self._generate_health_data(500)
        
        # Generate synthetic infrastructure complaints
        if not any('complaint' in key or 'infrastructure' in key for key in self.data.keys()):
            logger.info("Generating synthetic infrastructure complaints...")
            self.data['infrastructure_complaints'] = self._generate_infrastructure_complaints(1000)
        
        # Generate synthetic service requests
        if 'service_requests' not in self.data:
            logger.info("Generating synthetic service requests...")
            self.data['service_requests'] = self._generate_service_requests(2000)
        
        # Generate synthetic citizen feedback
        if not any('feedback' in key or 'sentiment' in key for key in self.data.keys()):
            logger.info("Generating synthetic citizen feedback...")
            self.data['citizen_feedback'] = self._generate_citizen_feedback(800)
    
    def _generate_health_data(self, n_records: int) -> pd.DataFrame:
        """Generate synthetic health infrastructure data"""
        np.random.seed(42)
        
        wards = [f'Ward-{i}' for i in range(1, 16)]
        
        data = {
            'ward': np.random.choice(wards, n_records),
            'hospitals': np.random.randint(1, 8, n_records),
            'beds': np.random.randint(20, 500, n_records),
            'doctors': np.random.randint(5, 100, n_records),
            'nurses': np.random.randint(10, 200, n_records),
            'ambulances': np.random.randint(2, 15, n_records),
            'patients_per_day': np.random.randint(50, 800, n_records),
            'emergency_cases': np.random.randint(5, 100, n_records),
            'occupancy_rate': np.random.uniform(0.4, 0.95, n_records),
            'avg_response_time_min': np.random.uniform(8, 45, n_records)
        }
        
        return pd.DataFrame(data)
    
    def _generate_infrastructure_complaints(self, n_records: int) -> pd.DataFrame:
        """Generate synthetic infrastructure complaints"""
        np.random.seed(43)
        
        locations = [f'Location-{i}' for i in range(1, 101)]
        issue_types = ['Pothole', 'Water Leakage', 'Street Light', 'Drainage', 'Road Damage']
        statuses = ['Open', 'In Progress', 'Resolved', 'Closed']
        priorities = ['Low', 'Medium', 'High', 'Critical']
        
        dates = pd.date_range(start='2023-01-01', end='2024-11-05', periods=n_records)
        
        data = {
            'complaint_id': [f'CMP-{i:06d}' for i in range(1, n_records + 1)],
            'date': dates,
            'location': np.random.choice(locations, n_records),
            'issue_type': np.random.choice(issue_types, n_records),
            'status': np.random.choice(statuses, n_records),
            'priority': np.random.choice(priorities, n_records),
            'response_time_hours': np.random.uniform(1, 72, n_records),
            'resolution_time_hours': np.random.uniform(2, 168, n_records),
            'citizen_satisfaction': np.random.randint(1, 6, n_records),
            'recurring': np.random.choice([0, 1], n_records, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def _generate_service_requests(self, n_records: int) -> pd.DataFrame:
        """Generate synthetic service requests"""
        np.random.seed(44)
        
        services = ['Healthcare', 'Water Supply', 'Electricity', 'Sanitation', 'Transport']
        departments = ['Health', 'PWD', 'Electricity Board', 'Sanitation Dept', 'Transport Dept']
        
        dates = pd.date_range(start='2023-01-01', end='2024-11-05', periods=n_records)
        
        data = {
            'request_id': [f'REQ-{i:06d}' for i in range(1, n_records + 1)],
            'date': dates,
            'hour': np.random.randint(0, 24, n_records),
            'day_of_week': dates.dayofweek,
            'service_type': np.random.choice(services, n_records),
            'department': np.random.choice(departments, n_records),
            'urgency': np.random.randint(1, 11, n_records),
            'processing_time_hours': np.random.uniform(0.5, 48, n_records),
            'resolved': np.random.choice([0, 1], n_records, p=[0.15, 0.85])
        }
        
        return pd.DataFrame(data)
    
    def _generate_citizen_feedback(self, n_records: int) -> pd.DataFrame:
        """Generate synthetic citizen feedback with sentiment"""
        np.random.seed(45)
        
        services = ['Healthcare', 'Roads', 'Water', 'Electricity', 'Sanitation']
        sentiments = ['Positive', 'Neutral', 'Negative']
        
        positive_comments = [
            'Excellent service', 'Very satisfied', 'Quick response', 
            'Good quality', 'Professional staff'
        ]
        neutral_comments = [
            'Average service', 'Could be better', 'Acceptable',
            'Normal experience', 'As expected'
        ]
        negative_comments = [
            'Poor service', 'Very disappointed', 'Slow response',
            'Bad quality', 'Unprofessional staff'
        ]
        
        data = {
            'feedback_id': [f'FBK-{i:06d}' for i in range(1, n_records + 1)],
            'date': pd.date_range(start='2023-01-01', end='2024-11-05', periods=n_records),
            'service': np.random.choice(services, n_records),
            'rating': np.random.randint(1, 6, n_records),
            'sentiment': np.random.choice(sentiments, n_records, p=[0.4, 0.3, 0.3]),
        }
        
        # Add comments based on sentiment
        comments = []
        for sentiment in data['sentiment']:
            if sentiment == 'Positive':
                comments.append(np.random.choice(positive_comments))
            elif sentiment == 'Neutral':
                comments.append(np.random.choice(neutral_comments))
            else:
                comments.append(np.random.choice(negative_comments))
        
        data['comment'] = comments
        
        return pd.DataFrame(data)
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get specific dataset by name"""
        return self.data.get(name)
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded datasets"""
        stats = {
            'total_datasets': len(self.data),
            'total_records': sum(len(df) for df in self.data.values()),
            'datasets': {}
        }
        
        for name, df in self.data.items():
            stats['datasets'][name] = {
                'records': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        return stats


if __name__ == "__main__":
    print("=" * 80)
    print("REAL DATA LOADER TEST")
    print("=" * 80)
    
    loader = RealDataLoader()
    data = loader.load_all_datasets()
    
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    stats = loader.get_statistics()
    print(f"\nTotal Datasets: {stats['total_datasets']}")
    print(f"Total Records: {stats['total_records']:,}")
    print(f"\nDataset Details:")
    for name, info in stats['datasets'].items():
        print(f"  {name}:")
        print(f"    Records: {info['records']:,}")
        print(f"    Columns: {info['columns']}")
        print(f"    Memory: {info['memory_mb']:.2f} MB")
    
    print("\n" + "=" * 80)
    print("DATA LOADER READY")
    print("=" * 80)
