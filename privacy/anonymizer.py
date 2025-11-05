"""
Data Anonymizer - PII detection and masking
"""

import re
import hashlib
import pandas as pd
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnonymizer:
    """Anonymize sensitive citizen data"""
    
    def __init__(self):
        self.token_map = {}
        self.salt = "governance_platform_salt_2024"
        
        self.pii_patterns = {
            'phone': r'\b\d{10}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'aadhaar': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
            'pan': r'\b[A-Z]{5}\d{4}[A-Z]\b',
        }
        
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text"""
        detected = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, str(text))
            for match in matches:
                detected.append({
                    'type': pii_type,
                    'value': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return detected
    
    def mask_text(self, text: str, mask_char: str = '*') -> str:
        """Mask PII in text"""
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        pii_items = self.detect_pii(text)
        
        masked_text = text
        offset = 0
        
        for pii in pii_items:
            start = pii['start'] + offset
            end = pii['end'] + offset
            original = pii['value']
            
            # Different masking strategies
            if pii['type'] == 'phone':
                masked = f"{original[:2]}{mask_char * 6}{original[-2:]}"
            elif pii['type'] == 'email':
                parts = original.split('@')
                masked = f"{parts[0][:2]}{mask_char * 4}@{parts[1]}"
            elif pii['type'] == 'aadhaar':
                masked = f"{mask_char * 8}{original[-4:]}"
            elif pii['type'] == 'pan':
                masked = f"{original[:3]}{mask_char * 6}{original[-1]}"
            else:
                masked = mask_char * len(original)
            
            masked_text = masked_text[:start] + masked + masked_text[end:]
            offset += len(masked) - len(original)
        
        return masked_text
    
    def anonymize_dataframe(self, df: pd.DataFrame, text_columns: List[str] = None) -> pd.DataFrame:
        """Anonymize DataFrame"""
        df_anon = df.copy()
        
        if text_columns is None:
            # Auto-detect text columns
            text_columns = df_anon.select_dtypes(include=['object']).columns.tolist()
        
        for col in text_columns:
            if col in df_anon.columns:
                df_anon[col] = df_anon[col].apply(self.mask_text)
        
        return df_anon
    
    def create_hash(self, value: str) -> str:
        """Create hash for value"""
        hash_input = f"{value}{self.salt}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def get_anonymization_stats(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict:
        """Get anonymization statistics"""
        stats = {
            'total_rows': len(original_df),
            'columns_processed': len(anonymized_df.columns),
            'pii_detected': 0
        }
        
        # Count PII instances
        for col in original_df.select_dtypes(include=['object']).columns:
            for text in original_df[col]:
                if isinstance(text, str):
                    stats['pii_detected'] += len(self.detect_pii(text))
        
        return stats


if __name__ == "__main__":
    print("=" * 80)
    print("DATA ANONYMIZER TEST")
    print("=" * 80)
    
    # Test data
    test_data = {
        'name': ['Rahul Sharma', 'Priya Patel', 'Amit Kumar'],
        'phone': ['9876543210', '9123456789', '8765432109'],
        'email': ['rahul@example.com', 'priya@example.com', 'amit@example.com'],
        'description': [
            'Please call me at 9876543210',
            'My email is priya@example.com for updates',
            'Contact 8765432109 or amit@example.com'
        ],
        'location': ['Pune', 'Mumbai', 'Nagpur']
    }
    
    df = pd.DataFrame(test_data)
    
    print("\nOriginal Data:")
    print(df)
    
    # Anonymize
    anonymizer = DataAnonymizer()
    df_anon = anonymizer.anonymize_dataframe(df)
    
    print("\n" + "=" * 80)
    print("Anonymized Data:")
    print(df_anon)
    
    # Stats
    print("\n" + "=" * 80)
    print("Anonymization Statistics:")
    stats = anonymizer.get_anonymization_stats(df, df_anon)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
    print("DATA ANONYMIZER READY")
    print("=" * 80)
