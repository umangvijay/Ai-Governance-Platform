"""
Multi-Sectoral Data Merger - Combine datasets from different sectors
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiSectoralMerger:
    """Merge and integrate multi-sectoral datasets"""
    
    def __init__(self):
        self.datasets = {}
        self.merged_data = None
        
    def add_dataset(self, name: str, df: pd.DataFrame, sector: str):
        """Add a dataset to the merger"""
        self.datasets[name] = {
            'data': df,
            'sector': sector,
            'records': len(df)
        }
        logger.info(f"Added dataset: {name} ({sector}) - {len(df)} records")
    
    def merge_by_location(self, location_col: str = 'district') -> pd.DataFrame:
        """Merge datasets by location (district/tehsil)"""
        logger.info("Merging datasets by location...")
        
        if not self.datasets:
            logger.error("No datasets to merge")
            return None
        
        # Start with first dataset
        merged = None
        
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data'].copy()
            
            # Standardize location column
            if location_col in df.columns:
                df['location'] = df[location_col]
            elif 'district' in df.columns:
                df['location'] = df['district']
            elif 'tehsil' in df.columns:
                df['location'] = df['tehsil']
            else:
                # Add default location
                df['location'] = 'Unknown'
            
            # Add sector prefix to columns
            sector = dataset_info['sector']
            df = df.add_prefix(f"{sector}_")
            df.rename(columns={f"{sector}_location": 'location'}, inplace=True)
            
            # Aggregate by location
            if 'location' in df.columns:
                df_agg = df.groupby('location').mean(numeric_only=True).reset_index()
            else:
                df_agg = df
            
            # Merge
            if merged is None:
                merged = df_agg
            else:
                merged = pd.merge(merged, df_agg, on='location', how='outer')
        
        self.merged_data = merged
        logger.info(f"✓ Merged {len(self.datasets)} datasets into {len(merged)} location records")
        
        return merged
    
    def merge_by_time(self, time_col: str = 'month') -> pd.DataFrame:
        """Merge datasets by time period"""
        logger.info("Merging datasets by time period...")
        
        merged = None
        
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data'].copy()
            
            # Standardize time column
            if time_col in df.columns:
                df['time_period'] = df[time_col]
            elif 'year' in df.columns and 'month' in df.columns:
                df['time_period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            elif 'year' in df.columns:
                df['time_period'] = df['year'].astype(str)
            else:
                df['time_period'] = 'Unknown'
            
            # Add sector prefix
            sector = dataset_info['sector']
            df = df.add_prefix(f"{sector}_")
            df.rename(columns={f"{sector}_time_period": 'time_period'}, inplace=True)
            
            # Aggregate by time
            df_agg = df.groupby('time_period').mean(numeric_only=True).reset_index()
            
            # Merge
            if merged is None:
                merged = df_agg
            else:
                merged = pd.merge(merged, df_agg, on='time_period', how='outer')
        
        self.merged_data = merged
        logger.info(f"✓ Merged {len(self.datasets)} datasets into {len(merged)} time periods")
        
        return merged
    
    def merge_spatial_temporal(self) -> pd.DataFrame:
        """Merge datasets by both location and time"""
        logger.info("Merging datasets by location and time...")
        
        merged = None
        
        for name, dataset_info in self.datasets.items():
            df = dataset_info['data'].copy()
            
            # Standardize columns
            if 'district' in df.columns:
                df['location'] = df['district']
            elif 'tehsil' in df.columns:
                df['location'] = df['tehsil']
            else:
                df['location'] = 'Unknown'
            
            if 'year' in df.columns and 'month' in df.columns:
                df['time_period'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            elif 'year' in df.columns:
                df['time_period'] = df['year'].astype(str)
            else:
                df['time_period'] = '2023-01'
            
            # Add sector prefix
            sector = dataset_info['sector']
            df = df.add_prefix(f"{sector}_")
            df.rename(columns={
                f"{sector}_location": 'location',
                f"{sector}_time_period": 'time_period'
            }, inplace=True)
            
            # Aggregate
            df_agg = df.groupby(['location', 'time_period']).mean(numeric_only=True).reset_index()
            
            # Merge
            if merged is None:
                merged = df_agg
            else:
                merged = pd.merge(merged, df_agg, on=['location', 'time_period'], how='outer')
        
        self.merged_data = merged
        logger.info(f"✓ Merged {len(self.datasets)} datasets into {len(merged)} spatial-temporal records")
        
        return merged
    
    def create_risk_score(self) -> pd.DataFrame:
        """Create composite risk score from merged data"""
        if self.merged_data is None:
            logger.error("No merged data available")
            return None
        
        df = self.merged_data.copy()
        
        # Calculate risk components
        risk_components = []
        
        # Health risk
        health_cols = [col for col in df.columns if 'health_' in col]
        if health_cols:
            df['health_risk'] = df[health_cols].fillna(0).mean(axis=1)
            risk_components.append('health_risk')
        
        # Crime risk
        crime_cols = [col for col in df.columns if 'crime_' in col or 'cyber_' in col]
        if crime_cols:
            df['crime_risk'] = df[crime_cols].fillna(0).mean(axis=1)
            risk_components.append('crime_risk')
        
        # Infrastructure risk
        infra_cols = [col for col in df.columns if 'prison_' in col]
        if infra_cols:
            df['infrastructure_risk'] = df[infra_cols].fillna(0).mean(axis=1)
            risk_components.append('infrastructure_risk')
        
        # Composite risk score
        if risk_components:
            df['composite_risk_score'] = df[risk_components].mean(axis=1)
            
            # Normalize to 0-100 scale
            if df['composite_risk_score'].max() > 0:
                df['composite_risk_score'] = (df['composite_risk_score'] / 
                                             df['composite_risk_score'].max() * 100)
        
        logger.info("✓ Created composite risk scores")
        return df
    
    def get_high_risk_locations(self, threshold: float = 70.0) -> pd.DataFrame:
        """Get locations with high risk scores"""
        if 'composite_risk_score' not in self.merged_data.columns:
            self.create_risk_score()
        
        high_risk = self.merged_data[
            self.merged_data['composite_risk_score'] >= threshold
        ].sort_values('composite_risk_score', ascending=False)
        
        logger.info(f"Found {len(high_risk)} high-risk locations (threshold: {threshold})")
        return high_risk
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix of merged data"""
        if self.merged_data is None:
            logger.error("No merged data available")
            return None
        
        numeric_cols = self.merged_data.select_dtypes(include=[np.number]).columns
        correlation = self.merged_data[numeric_cols].corr()
        
        logger.info("✓ Generated correlation matrix")
        return correlation


if __name__ == "__main__":
    print("=" * 80)
    print("MULTI-SECTORAL DATA MERGER TEST")
    print("=" * 80)
    
    # Create sample datasets
    health_data = pd.DataFrame({
        'district': ['Pune', 'Mumbai', 'Thane'] * 10,
        'beds': np.random.randint(100, 1000, 30),
        'patients': np.random.randint(50, 800, 30),
        'year': [2023] * 30,
        'month': np.random.randint(1, 13, 30)
    })
    
    crime_data = pd.DataFrame({
        'district': ['Pune', 'Mumbai', 'Thane'] * 10,
        'crime_cases': np.random.randint(10, 200, 30),
        'solved_cases': np.random.randint(5, 150, 30),
        'year': [2023] * 30,
        'month': np.random.randint(1, 13, 30)
    })
    
    prison_data = pd.DataFrame({
        'district': ['Pune', 'Mumbai', 'Thane'] * 10,
        'capacity': np.random.randint(100, 2000, 30),
        'inmates': np.random.randint(80, 2200, 30),
        'year': [2023] * 30
    })
    
    # Create merger
    merger = MultiSectoralMerger()
    merger.add_dataset('health', health_data, 'health')
    merger.add_dataset('crime', crime_data, 'crime')
    merger.add_dataset('prison', prison_data, 'prison')
    
    # Test different merge strategies
    print("\n### MERGE BY LOCATION ###")
    location_merged = merger.merge_by_location()
    print(f"Merged data shape: {location_merged.shape}")
    print(f"Locations: {location_merged['location'].unique()}")
    
    print("\n### CREATE RISK SCORES ###")
    risk_data = merger.create_risk_score()
    print("\nComposite Risk Scores:")
    print(risk_data[['location', 'composite_risk_score']].head(10))
    
    print("\n### HIGH RISK LOCATIONS ###")
    high_risk = merger.get_high_risk_locations(threshold=50.0)
    print(f"High risk locations: {len(high_risk)}")
    if len(high_risk) > 0:
        print(high_risk[['location', 'composite_risk_score']].head())
    
    print("\n" + "=" * 80)
    print("MULTI-SECTORAL MERGER READY")
    print("=" * 80)
