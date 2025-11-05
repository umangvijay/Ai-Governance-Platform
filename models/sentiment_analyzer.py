"""
Sentiment Analyzer - Analyzes citizen feedback sentiment
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from citizen feedback"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
        self.metrics = {}
        self.sentiment_labels = ['Negative', 'Neutral', 'Positive']
        
    def train(self, X: pd.DataFrame, y: pd.Series, text_column: str = 'comment'):
        """Train sentiment analysis model"""
        logger.info("Training Sentiment Analyzer...")
        
        # Extract text
        if text_column not in X.columns:
            logger.warning(f"Text column '{text_column}' not found, using numeric features only")
            X_features = X
        else:
            # Vectorize text
            X_text = self.vectorizer.fit_transform(X[text_column].fillna(''))
            X_text_df = pd.DataFrame(
                X_text.toarray(),
                columns=[f'word_{i}' for i in range(X_text.shape[1])]
            )
            
            # Combine with numeric features
            numeric_cols = [col for col in X.columns if col != text_column and X[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                X_features = pd.concat([X[numeric_cols].reset_index(drop=True), X_text_df], axis=1)
            else:
                X_features = X_text_df
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        
        self.metrics = {
            'accuracy': acc,
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        logger.info(f"  Accuracy: {acc:.4f}")
        
        self.is_trained = True
        logger.info("✓ Sentiment Analyzer trained successfully")
        
        return self.metrics
    
    def predict(self, X: pd.DataFrame, text_column: str = 'comment') -> np.ndarray:
        """Predict sentiment"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract features
        if text_column in X.columns:
            X_text = self.vectorizer.transform(X[text_column].fillna(''))
            X_text_df = pd.DataFrame(
                X_text.toarray(),
                columns=[f'word_{i}' for i in range(X_text.shape[1])]
            )
            
            numeric_cols = [col for col in X.columns if col != text_column and X[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                X_features = pd.concat([X[numeric_cols].reset_index(drop=True), X_text_df], axis=1)
            else:
                X_features = X_text_df
        else:
            X_features = X
        
        return self.model.predict(X_features)
    
    def predict_proba(self, X: pd.DataFrame, text_column: str = 'comment') -> np.ndarray:
        """Predict sentiment probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract features (same as predict)
        if text_column in X.columns:
            X_text = self.vectorizer.transform(X[text_column].fillna(''))
            X_text_df = pd.DataFrame(
                X_text.toarray(),
                columns=[f'word_{i}' for i in range(X_text.shape[1])]
            )
            
            numeric_cols = [col for col in X.columns if col != text_column and X[col].dtype in ['int64', 'float64']]
            if numeric_cols:
                X_features = pd.concat([X[numeric_cols].reset_index(drop=True), X_text_df], axis=1)
            else:
                X_features = X_text_df
        else:
            X_features = X
        
        return self.model.predict_proba(X_features)
    
    def analyze_sentiment_distribution(self, predictions: np.ndarray) -> dict:
        """Analyze sentiment distribution"""
        unique, counts = np.unique(predictions, return_counts=True)
        total = len(predictions)
        
        distribution = {}
        for sentiment, count in zip(unique, counts):
            distribution[sentiment] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        return distribution
    
    def get_top_words(self, sentiment_class: str, top_n: int = 10) -> list:
        """Get top words for a sentiment class"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Get word features (first N features from vectorizer)
        n_words = len(self.vectorizer.get_feature_names_out())
        word_importances = importances[:n_words] if len(importances) >= n_words else importances
        
        # Get top words
        word_names = self.vectorizer.get_feature_names_out()
        top_indices = np.argsort(word_importances)[-top_n:][::-1]
        
        return [(word_names[i], word_importances[i]) for i in top_indices if i < len(word_names)]
    
    def save_model(self, path: str = 'models/saved_models/sentiment_analyzer.pkl'):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'sentiment_labels': self.sentiment_labels,
            'metrics': self.metrics,
            'is_trained': self.is_trained
        }, path)
        logger.info(f"✓ Model saved to {path}")
    
    def load_model(self, path: str = 'models/saved_models/sentiment_analyzer.pkl'):
        """Load trained model"""
        data = joblib.load(path)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.sentiment_labels = data['sentiment_labels']
        self.metrics = data['metrics']
        self.is_trained = data['is_trained']
        logger.info(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 80)
    print("SENTIMENT ANALYZER TEST")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 800
    
    sentiments = ['Positive', 'Neutral', 'Negative']
    
    positive_comments = [
        'Excellent service quality', 'Very satisfied with response', 'Quick and professional',
        'Good infrastructure', 'Well maintained roads', 'Clean and efficient',
        'Highly recommend', 'Great improvement', 'Fantastic work'
    ]
    neutral_comments = [
        'Average service', 'Could be better', 'Acceptable quality',
        'Normal experience', 'As expected', 'Standard service',
        'Moderate satisfaction', 'Decent work', 'Fair response'
    ]
    negative_comments = [
        'Poor service quality', 'Very disappointed', 'Slow response time',
        'Bad infrastructure', 'Poorly maintained', 'Dirty and inefficient',
        'Not recommended', 'Needs improvement', 'Terrible experience'
    ]
    
    comments = []
    sentiment_labels = []
    ratings = []
    
    for _ in range(n_samples):
        sentiment = np.random.choice(sentiments, p=[0.4, 0.3, 0.3])
        
        if sentiment == 'Positive':
            comment = np.random.choice(positive_comments)
            rating = np.random.randint(4, 6)
        elif sentiment == 'Neutral':
            comment = np.random.choice(neutral_comments)
            rating = np.random.randint(3, 4)
        else:
            comment = np.random.choice(negative_comments)
            rating = np.random.randint(1, 3)
        
        comments.append(comment)
        sentiment_labels.append(sentiment)
        ratings.append(rating)
    
    X = pd.DataFrame({
        'comment': comments,
        'rating': ratings,
        'service': np.random.choice(['Healthcare', 'Roads', 'Water', 'Electricity'], n_samples)
    })
    
    y = pd.Series(sentiment_labels)
    
    # Train model
    analyzer = SentimentAnalyzer()
    metrics = analyzer.train(X, y, text_column='comment')
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Test predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    sample = X.head(10)
    predictions = analyzer.predict(sample, text_column='comment')
    probabilities = analyzer.predict_proba(sample, text_column='comment')
    
    print("\nSentiment Predictions:")
    for i, (comment, pred, proba) in enumerate(zip(sample['comment'], predictions, probabilities)):
        max_prob = np.max(proba)
        print(f"\n  Sample {i+1}:")
        print(f"    Comment: '{comment}'")
        print(f"    Sentiment: {pred} (Confidence: {max_prob:.2%})")
    
    # Sentiment distribution
    print("\n" + "=" * 80)
    print("SENTIMENT DISTRIBUTION")
    print("=" * 80)
    
    all_predictions = analyzer.predict(X, text_column='comment')
    distribution = analyzer.analyze_sentiment_distribution(all_predictions)
    
    print("\nOverall Sentiment:")
    for sentiment, stats in distribution.items():
        print(f"  {sentiment}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # Top words
    print("\n" + "=" * 80)
    print("TOP INFLUENTIAL WORDS")
    print("=" * 80)
    
    try:
        top_words = analyzer.get_top_words('Positive', top_n=10)
        print("\nTop 10 Words:")
        for word, importance in top_words:
            print(f"  {word}: {importance:.4f}")
    except Exception as e:
        logger.warning(f"Could not extract top words: {e}")
    
    # Save model
    analyzer.save_model()
    
    print("\n" + "=" * 80)
    print("SENTIMENT ANALYZER READY")
    print("=" * 80)
