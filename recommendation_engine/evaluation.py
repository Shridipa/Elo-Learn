"""Recommendation Evaluation Metrics

Implements evaluation metrics for recommendation systems:
- Precision@K
- Recall@K
- NDCG (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Coverage
- Diversity
"""

import numpy as np
from typing import List, Dict, Set
import logging

logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """Compute recommendation system evaluation metrics"""
    
    @staticmethod
    def precision_at_k(relevant_items: Set, recommended_items: List, k: int) -> float:
        """
        Precision@K: Fraction of recommended items that are relevant
        
        P@K = |recommended@K ∩ relevant| / K
        """
        if k <= 0:
            return 0.0
        
        recommended_at_k = set(recommended_items[:k])
        intersection = len(recommended_at_k & relevant_items)
        return intersection / k
    
    @staticmethod
    def recall_at_k(relevant_items: Set, recommended_items: List, k: int) -> float:
        """
        Recall@K: Fraction of relevant items that are recommended
        
        R@K = |recommended@K ∩ relevant| / |relevant|
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_at_k = set(recommended_items[:k])
        intersection = len(recommended_at_k & relevant_items)
        return intersection / len(relevant_items)
    
    @staticmethod
    def ndcg_at_k(relevant_items: Set, recommended_items: List, k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Measures ranking quality considering position
        DCG = Σ(rel_i / log2(i+1)) for i in 1..k
        IDCG = DCG of ideal ranking
        NDCG = DCG / IDCG
        """
        if k <= 0 or len(relevant_items) == 0:
            return 0.0
        
        # Compute DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)
        
        # Compute ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_items))))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mean_average_precision_at_k(relevant_items_list: List[Set],
                                   recommended_items_list: List[List],
                                   k: int) -> float:
        """
        MAP@K: Mean Average Precision
        
        Average of precision values at each relevant item position
        """
        aps = []
        
        for relevant_items, recommended_items in zip(relevant_items_list, recommended_items_list):
            if len(relevant_items) == 0:
                aps.append(0.0)
                continue
            
            precisions = []
            for i, item in enumerate(recommended_items[:k]):
                if item in relevant_items:
                    p_at_i = RecommendationMetrics.precision_at_k(relevant_items, recommended_items, i + 1)
                    precisions.append(p_at_i)
            
            if precisions:
                aps.append(np.mean(precisions))
            else:
                aps.append(0.0)
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def coverage(all_recommended_items: List, all_items: Set) -> float:
        """
        Coverage: Fraction of items that are recommended at least once
        
        Measures diversity of recommendations
        """
        unique_recommended = set()
        for items in all_recommended_items:
            unique_recommended.update(items)
        
        return len(unique_recommended) / len(all_items) if len(all_items) > 0 else 0.0
    
    @staticmethod
    def diversity(recommendations_list: List[List[str]]) -> float:
        """
        Diversity: Average dissimilarity between items in recommendations
        
        Simplified: Fraction of items that appear only once
        """
        from collections import Counter
        
        all_items = []
        for recs in recommendations_list:
            all_items.extend(recs)
        
        if len(all_items) == 0:
            return 0.0
        
        counter = Counter(all_items)
        unique_count = sum(1 for count in counter.values() if count == 1)
        
        return unique_count / len(all_items)


class OfflineEvaluator:
    """Offline evaluation framework using historical data"""
    
    def __init__(self, test_size: float = 0.2, random_seed: int = 42):
        self.test_size = test_size
        self.random_seed = random_seed
    
    @staticmethod
    def evaluate_recommender(recommender,
                            test_interactions,
                            top_k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Evaluate recommender on test interactions
        
        Returns:
            Dictionary of metrics for each k value
        """
        logger.info(f"Evaluating recommender on {len(test_interactions)} test samples...")
        
        results = {}
        
        for k in top_k_values:
            precisions = []
            recalls = []
            ndcgs = []
            
            students = test_interactions['student_id'].unique()
            
            for student_id in students:
                student_data = test_interactions[test_interactions['student_id'] == student_id]
                
                if len(student_data) == 0:
                    continue
                
                # Get all topics
                all_topics = list(student_data['topic'].unique())
                
                # Get recommendations
                try:
                    recs = recommender.recommend(student_id, all_topics, top_k=k)
                    recommended_topics = [r['topic'] for r in recs]
                except Exception as e:
                    logger.warning(f"Error getting recommendations for student {student_id}: {e}")
                    continue
                
                # Relevant items: those the student attempted AND succeeded at
                relevant_items = set(
                    student_data[student_data['score'] > 0.7]['topic'].values
                )
                
                if len(relevant_items) == 0:
                    relevant_items = set(student_data['topic'].values)
                
                # Compute metrics
                precision = RecommendationMetrics.precision_at_k(
                    relevant_items, recommended_topics, k
                )
                recall = RecommendationMetrics.recall_at_k(
                    relevant_items, recommended_topics, k
                )
                ndcg = RecommendationMetrics.ndcg_at_k(
                    relevant_items, recommended_topics, k
                )
                
                precisions.append(precision)
                recalls.append(recall)
                ndcgs.append(ndcg)
            
            results[f'precision@{k}'] = np.mean(precisions) if precisions else 0.0
            results[f'recall@{k}'] = np.mean(recalls) if recalls else 0.0
            results[f'ndcg@{k}'] = np.mean(ndcgs) if ndcgs else 0.0
        
        logger.info("✓ Evaluation complete")
        logger.info(f"Results: {results}")
        
        return results
