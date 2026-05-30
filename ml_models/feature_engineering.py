"""Feature Engineering Pipeline for Elo Learn

Transforms raw student interaction data into feature vectors for ML models.
This is critical for recommendation systems and RL agents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for students and topics"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
    
    @staticmethod
    def aggregate_student_features(interactions: pd.DataFrame, 
                                  student_id: int) -> Dict[str, float]:
        """
        Aggregate features for a specific student
        
        Features include:
        - Performance metrics
        - Time patterns
        - Learning speed
        - Consistency
        - Engagement
        """
        student_data = interactions[interactions['student_id'] == student_id]
        
        if len(student_data) == 0:
            return {}
        
        features = {
            # Performance metrics
            'success_rate': student_data['is_correct'].mean(),
            'avg_score': student_data['score'].mean(),
            'score_std': student_data['score'].std() if len(student_data) > 1 else 0,
            
            # Time metrics
            'avg_time_spent': student_data['time_spent'].mean(),
            'median_time_spent': student_data['time_spent'].median(),
            'time_variance': student_data['time_spent'].var() if len(student_data) > 1 else 0,
            
            # Learning speed (time improvement with repetition)
            'learning_speed': FeatureEngineer._calculate_learning_speed(student_data),
            
            # Engagement
            'engagement_score': FeatureEngineer._calculate_engagement(student_data),
            
            # Consistency
            'recent_performance': student_data['score'].tail(10).mean(),
            'performance_trend': FeatureEngineer._calculate_trend(student_data['score']),
            
            # Difficulty adaptation
            'preferred_difficulty': student_data['difficulty_presented'].mean(),
            'difficulty_progression': FeatureEngineer._calculate_trend(
                student_data['difficulty_presented']
            ),
        }
        
        return features
    
    @staticmethod
    def aggregate_topic_features(interactions: pd.DataFrame,
                                topic: str) -> Dict[str, float]:
        """Aggregate features for a specific topic"""
        topic_data = interactions[interactions['topic'] == topic]
        
        if len(topic_data) == 0:
            return {}
        
        features = {
            # Difficulty metrics
            'success_rate': topic_data['is_correct'].mean(),
            'avg_difficulty_presented': topic_data['difficulty_presented'].mean(),
            'difficulty_std': topic_data['difficulty_presented'].std() if len(topic_data) > 1 else 0,
            
            # Time metrics
            'avg_time_spent': topic_data['time_spent'].mean(),
            'avg_attempts': topic_data['attempts'].mean() if 'attempts' in topic_data.columns else 1.0,
            
            # Popularity and engagement
            'unique_students': topic_data['student_id'].nunique(),
            'total_attempts': len(topic_data),
            
            # Trend
            'performance_trend': FeatureEngineer._calculate_trend(topic_data['score']),
        }
        
        return features
    
    @staticmethod
    def _calculate_learning_speed(data: pd.DataFrame) -> float:
        """Calculate how much time improvement with repetition"""
        if len(data) < 2:
            return 0.0
        
        time_data = data['time_spent'].values
        # Linear regression of time vs. attempt number
        x = np.arange(len(time_data))
        if np.std(x) > 0:
            slope = np.cov(x, time_data)[0, 1] / np.var(x)
            return -slope / np.mean(time_data)  # Negative means improvement
        return 0.0
    
    @staticmethod
    def _calculate_engagement(data: pd.DataFrame) -> float:
        """Calculate engagement score (frequency, consistency)"""
        if len(data) < 2:
            return 0.5
        
        # Time between interactions (smaller is more engaged)
        timestamps = pd.to_datetime(data['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds().values[1:]
        
        if len(time_diffs) > 0:
            avg_gap = np.mean(time_diffs)
            # Convert to engagement: frequent interactions = high engagement
            engagement = 1 / (1 + avg_gap / 86400)  # Normalize by 1 day
        else:
            engagement = 0.5
        
        return np.clip(engagement, 0, 1)
    
    @staticmethod
    def _calculate_trend(values: pd.Series) -> float:
        """Calculate trend in a series (improvement/decline over time)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = values.values
        
        if np.std(x) > 0:
            slope = np.cov(x, y)[0, 1] / np.var(x)
            # Normalize by range
            range_y = np.max(y) - np.min(y) if np.max(y) > np.min(y) else 1
            return slope / range_y
        
        return 0.0
    
    def create_student_embeddings(self, interactions: pd.DataFrame) -> Dict[int, np.ndarray]:
        """Create feature embeddings for all students"""
        embeddings = {}
        
        for student_id in interactions['student_id'].unique():
            features = self.aggregate_student_features(interactions, student_id)
            
            # Convert to embedding vector
            feature_keys = sorted(features.keys())
            feature_values = np.array([features[k] for k in feature_keys])
            
            # Normalize features
            if np.any(np.isnan(feature_values)):
                feature_values = np.nan_to_num(feature_values)
            
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            if std_val > 0:
                feature_values = (feature_values - mean_val) / std_val
            
            embeddings[student_id] = feature_values
        
        return embeddings
    
    def create_topic_embeddings(self, interactions: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create feature embeddings for all topics"""
        embeddings = {}
        
        for topic in interactions['topic'].unique():
            features = self.aggregate_topic_features(interactions, topic)
            
            feature_keys = sorted(features.keys())
            feature_values = np.array([features[k] for k in feature_keys])
            
            if np.any(np.isnan(feature_values)):
                feature_values = np.nan_to_num(feature_values)
            
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            if std_val > 0:
                feature_values = (feature_values - mean_val) / std_val
            
            embeddings[topic] = feature_values
        
        return embeddings


class InteractionFeatures:
    """Extract features from individual student-topic interactions"""
    
    @staticmethod
    def get_interaction_features(interaction: Dict) -> Dict[str, float]:
        """Extract features from a single interaction"""
        return {
            'is_correct': float(interaction['is_correct']),
            'score': interaction['score'],
            'time_spent_log': np.log1p(interaction['time_spent']),
            'difficulty_presented': interaction['difficulty_presented'],
            'attempts': interaction.get('attempts', 1.0),
        }


class SequenceFeatures:
    """Extract sequence-based features for sequential recommendation models"""
    
    @staticmethod
    def get_student_sequence(interactions: pd.DataFrame, 
                            student_id: int,
                            max_length: int = 50) -> List[str]:
        """Get sequence of topics attempted by student"""
        student_data = interactions[interactions['student_id'] == student_id].sort_values('timestamp')
        sequence = student_data['topic'].tail(max_length).tolist()
        return sequence
    
    @staticmethod
    def get_student_performance_sequence(interactions: pd.DataFrame,
                                        student_id: int,
                                        max_length: int = 50) -> List[float]:
        """Get sequence of performance scores"""
        student_data = interactions[interactions['student_id'] == student_id].sort_values('timestamp')
        sequence = student_data['score'].tail(max_length).tolist()
        return sequence


def engineer_features(interactions_df: pd.DataFrame) -> Tuple[Dict, Dict, Dict]:
    """
    Main feature engineering pipeline
    
    Returns:
        Tuple of (student_embeddings, topic_embeddings, interaction_features)
    """
    logger.info("Starting feature engineering pipeline...")
    
    engineer = FeatureEngineer(embedding_dim=64)
    
    # Create embeddings
    student_embeddings = engineer.create_student_embeddings(interactions_df)
    topic_embeddings = engineer.create_topic_embeddings(interactions_df)
    
    logger.info(f"✓ Created embeddings for {len(student_embeddings)} students")
    logger.info(f"✓ Created embeddings for {len(topic_embeddings)} topics")
    
    # Create feature matrix for interactions
    interaction_features = {}
    for idx, row in interactions_df.iterrows():
        features = InteractionFeatures.get_interaction_features(row.to_dict())
        interaction_features[idx] = features
    
    logger.info(f"✓ Extracted features for {len(interaction_features)} interactions")
    
    return student_embeddings, topic_embeddings, interaction_features
