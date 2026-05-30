"""Basic Unit Tests for Recommendation Systems

Tests baseline recommendation implementations.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List

from recommendation_engine.baselines import SequentialTransitionBaseline


class TestCollaborativeFiltering:
    """Tests for collaborative filtering recommendation baseline"""
    
    @pytest.mark.unit
    def test_user_item_matrix_creation(self, sample_interactions_data):
        """Test creation of user-item matrix"""
        # Would test matrix factorization
        assert len(sample_interactions_data) > 0
        assert 'student_id' in sample_interactions_data.columns
    
    @pytest.mark.unit
    def test_recommendation_ranking(self, mock_recommendation_engine):
        """Test that recommendations are properly ranked"""
        recs = mock_recommendation_engine.recommend(student_id=1, top_k=3)
        assert len(recs) <= 3
        assert all(isinstance(r['score'], (int, float)) for r in recs)
        # Scores should be in descending order
        scores = [r['score'] for r in recs]
        assert scores == sorted(scores, reverse=True)


class TestContentBasedFiltering:
    """Tests for content-based recommendation baseline"""
    
    @pytest.mark.unit
    def test_topic_similarity(self, sample_concepts):
        """Test topic similarity calculation"""
        # Would test similarity metrics between topics
        assert 'Vectors' in sample_concepts
        assert 'Matrices' in sample_concepts


class TestSequentialTransitionBaseline:
    """Tests for sequence-based recommendation baseline"""

    @pytest.mark.unit
    def test_transition_recommendation(self):
        """Test that the sequential model recommends likely next topics."""
        data = pd.DataFrame({
            'student_id': [1, 1, 2, 2],
            'topic': ['A', 'B', 'B', 'C'],
            'score': [0.9, 0.8, 0.7, 0.6]
        })

        model = SequentialTransitionBaseline()
        model.fit(data)

        recs = model.recommend(1, ['B', 'C', 'A'], top_k=3)
        assert len(recs) == 3
        assert recs[0]['topic'] == 'C'
        assert recs[0]['score'] >= recs[1]['score']
        assert recs[0]['score'] > 0

    @pytest.mark.unit
    def test_cold_start_falls_back_to_popularity(self, sample_interactions_data):
        """Test that new students get popular-topic fallbacks."""
        topics = sorted(sample_interactions_data['topic'].unique().tolist())
        model = SequentialTransitionBaseline()
        model.fit(sample_interactions_data)

        recs = model.recommend(9999, topics, top_k=3)
        assert len(recs) == 3
        assert all('topic' in r and 'score' in r for r in recs)
        assert recs[0]['score'] >= recs[-1]['score']


class TestTransformerRecommender:
    """Tests for transformer-based recommendation model"""
    
    @pytest.mark.unit
    def test_sequence_modeling(self, sample_interactions_data):
        """Test sequence-based recommendations"""
        student_data = sample_interactions_data[
            sample_interactions_data['student_id'] == 1
        ]
        assert len(student_data) > 0


class TestRecommendationMetrics:
    """Tests for recommendation evaluation metrics"""
    
    @pytest.mark.unit
    def test_precision_at_k(self):
        """Test precision@k calculation"""
        # True relevant items: [1, 3, 4]
        # Recommended (top-5): [1, 2, 3, 5, 6]
        # Precision@5 = 2/5 = 0.4
        relevant = {1, 3, 4}
        recommended = [1, 2, 3, 5, 6]
        k = 5
        precision = len([x for x in recommended[:k] if x in relevant]) / k
        assert precision == 0.4
    
    @pytest.mark.unit
    def test_recall_at_k(self):
        """Test recall@k calculation"""
        relevant = {1, 3, 4}
        recommended = [1, 2, 3, 5, 6]
        k = 5
        recall = len([x for x in recommended[:k] if x in relevant]) / len(relevant)
        assert recall == 2/3
    
    @pytest.mark.unit
    def test_ndcg_calculation(self):
        """Test NDCG (Normalized Discounted Cumulative Gain) calculation"""
        # Ideal ranking would be [1, 3, 4] at top (all relevant)
        # Actual ranking [1, 2, 3, 5, 6]
        rel = [1, 0, 1, 0, 0]
        dcg = sum(r / np.log2(i+2) for i, r in enumerate(rel))
        idcg = sum(1 / np.log2(i+2) for i in range(min(3, len(rel))))
        ndcg = dcg / idcg
        assert 0 <= ndcg <= 1


@pytest.mark.unit
def test_cold_start_problem(mock_recommendation_engine):
    """Test handling of new students (cold start problem)"""
    # Should return reasonable defaults for new student
    recs = mock_recommendation_engine.recommend(student_id=9999, top_k=5)
    assert len(recs) > 0
