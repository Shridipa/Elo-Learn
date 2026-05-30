"""Baseline Recommendation Models

Implements three baseline recommendation approaches:
1. Collaborative Filtering (Matrix Factorization)
2. Content-Based Filtering
3. Sequence-Based Transition Model
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)

class PopularityBaseline:
    """
    Popularity-based recommendation baseline.

    Approach: Recommend the most frequently interacted topics across all students.
    This is a simple research baseline for comparison with more advanced methods.
    """

    def __init__(self):
        self.topic_counts = Counter()
        self.topic_rank: List[str] = []
        self.total = 0

    def fit(self, interactions: pd.DataFrame) -> None:
        self.topic_counts = Counter(interactions['topic'].dropna().tolist())
        self.topic_rank = [topic for topic, _ in self.topic_counts.most_common()]
        self.total = sum(self.topic_counts.values()) or 1

    def recommend(self, student_id: int, topics: List[str], top_k: int = 5) -> List[Dict]:
        ranked = [topic for topic in self.topic_rank if topic in topics]
        ranked += [topic for topic in topics if topic not in ranked]
        recs = []
        for topic in ranked[:top_k]:
            score = float(self.topic_counts.get(topic, 0)) / float(self.total)
            recs.append({"topic": topic, "score": score})
        return recs

    def explain_v2(self, student_id: int, interactions_df, mastery, kg, student_embeddings=None, topic_embeddings=None, top_k: int = 5, **kwargs):
        return [
            {"topic": rec['topic'], "recommendation_score": rec['score']}
            for rec in self.recommend(student_id, sorted(interactions_df['topic'].dropna().unique().tolist()), top_k=top_k)
        ]


class CollaborativeFilteringBaseline:
    """
    Collaborative Filtering using Matrix Factorization (SVD)
    
    Approach: Factorize user-item (student-topic) matrix
    - Fast to train and inference
    - Good baseline
    - Struggles with new users/items (cold start)
    """
    
    def __init__(self, n_factors: int = 20, random_seed: int = 42):
        self.n_factors = n_factors
        self.random_seed = random_seed
        self.svd = None
        self.user_factors = None
        self.item_factors = None
        self.mean_rating = None
        self.scaler = StandardScaler()
        
    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Fit collaborative filtering model
        
        Args:
            interactions: DataFrame with columns [student_id, topic, score]
        """
        logger.info(f"Training Collaborative Filtering with {self.n_factors} factors...")
        
        # Create user-item matrix
        user_item_matrix = interactions.pivot_table(
            index='student_id',
            columns='topic',
            values='score',
            aggfunc='mean',
            fill_value=0
        )
        
        self.mean_rating = user_item_matrix.mean().mean()
        
        # Normalize
        user_item_matrix = user_item_matrix.fillna(self.mean_rating)
        user_item_matrix_normalized = user_item_matrix - self.mean_rating
        
        # SVD factorization - adjust n_components to not exceed matrix dimensions
        if user_item_matrix_normalized.shape[1] < 2:
            logger.info('Using fallback linear projection for single-topic CF matrix')
            self.svd = None
            self.user_factors = user_item_matrix_normalized.values.reshape(-1, 1)
            self.item_factors = np.ones((1, 1))
        else:
            max_components = max(1, user_item_matrix_normalized.shape[1] - 1)
            n_components = min(self.n_factors, max_components)
            n_components = min(n_components, user_item_matrix_normalized.shape[1])
            logger.info(f"Using {n_components} SVD components (limited by {user_item_matrix_normalized.shape[1]} topics)")
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_seed)
            self.user_factors = self.svd.fit_transform(user_item_matrix_normalized)
            self.item_factors = self.svd.components_.T
        
        # Store for inference
        self.student_ids = list(user_item_matrix.index)
        self.topics = list(user_item_matrix.columns)
        self.topic_to_idx = {topic: idx for idx, topic in enumerate(self.topics)}
        
        if self.svd is not None:
            logger.info(f"✓ CF Model trained. Explained variance: {self.svd.explained_variance_ratio_.sum():.4f}")
        else:
            logger.info(f"✓ CF Model trained (single-topic fallback)")
    
    def predict(self, student_id: int, topic: str) -> float:
        """Predict score for student-topic pair"""
        try:
            student_idx = self.student_ids.index(student_id)
            topic_idx = self.topic_to_idx[topic]
            
            score = np.dot(self.user_factors[student_idx], self.item_factors[topic_idx])
            score = np.clip(score + self.mean_rating, 0, 1)
            return float(score)
        except (ValueError, KeyError):
            return self.mean_rating
    
    def recommend(self, student_id: int, topics: List[str], top_k: int = 5) -> List[Dict]:
        """Get top-k recommendations for a student"""
        scores = []
        for topic in topics:
            score = self.predict(student_id, topic)
            scores.append((topic, score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"topic": topic, "score": score}
            for topic, score in scores[:top_k]
        ]

    def explain_v2(self, student_id: int, interactions_df, mastery, kg, student_embeddings=None, topic_embeddings=None, top_k: int = 5, **kwargs):
        topics = sorted(interactions_df['topic'].dropna().unique().tolist())
        recs = self.recommend(student_id, topics, top_k=top_k)
        return [{"topic": rec['topic'], "recommendation_score": rec['score']} for rec in recs]


class ContentBasedFilteringBaseline:
    """
    Content-Based Filtering
    
    Approach: Recommend topics similar to those student liked
    - Handles cold start better
    - Explainable
    - May be limited to similar content
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.student_profiles = {}
        self.topic_embeddings = {}
        
    def fit(self, interactions: pd.DataFrame, topic_embeddings: Dict[str, np.ndarray]) -> None:
        """
        Fit content-based model
        
        Args:
            interactions: DataFrame with student interactions
            topic_embeddings: Pre-computed topic embeddings
        """
        logger.info("Training Content-Based Filtering...")
        
        self.topic_embeddings = topic_embeddings
        
        # Build student profiles based on topics they liked
        for student_id in interactions['student_id'].unique():
            student_data = interactions[interactions['student_id'] == student_id]
            
            # Weighted average of topic embeddings (weighted by score)
            liked_topics = student_data[student_data['score'] > 0.7]['topic'].values
            liked_scores = student_data[student_data['score'] > 0.7]['score'].values
            
            if len(liked_topics) > 0:
                # Get embeddings for liked topics
                embeddings = []
                weights = []
                for topic, score in zip(liked_topics, liked_scores):
                    if topic in topic_embeddings:
                        embeddings.append(topic_embeddings[topic])
                        weights.append(score)
                
                if embeddings:
                    # Weighted average
                    weights = np.array(weights) / np.sum(weights)
                    self.student_profiles[student_id] = np.average(embeddings, axis=0, weights=weights)
                else:
                    # Default profile
                    self.student_profiles[student_id] = np.mean(list(topic_embeddings.values()), axis=0)
            else:
                self.student_profiles[student_id] = np.mean(list(topic_embeddings.values()), axis=0)
        
        logger.info(f"✓ Content-Based model trained for {len(self.student_profiles)} students")
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return np.dot(v1, v2) / (norm1 * norm2)
        return 0.0
    
    def recommend(self, student_id: int, topics: List[str], top_k: int = 5) -> List[Dict]:
        """Get top-k recommendations based on content similarity"""
        if student_id not in self.student_profiles:
            # Default: return random topics
            return [{"topic": t, "score": 0.5} for t in topics[:top_k]]
        
        student_profile = self.student_profiles[student_id]
        scores = []
        
        for topic in topics:
            if topic in self.topic_embeddings:
                similarity = self._cosine_similarity(
                    student_profile,
                    self.topic_embeddings[topic]
                )
                scores.append((topic, similarity))
            else:
                scores.append((topic, 0.0))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"topic": topic, "score": max(0, score)}
            for topic, score in scores[:top_k]
        ]

    def explain_v2(self, student_id: int, interactions_df, mastery, kg, student_embeddings=None, topic_embeddings=None, top_k: int = 5, **kwargs):
        topics = sorted(interactions_df['topic'].dropna().unique().tolist())
        recs = self.recommend(student_id, topics, top_k=top_k)
        return [{"topic": rec['topic'], "recommendation_score": rec['score']} for rec in recs]


class SequentialTransitionBaseline:
    """
    Sequence-based recommendation baseline

    Approach: Recommend likely next topics based on topic transitions
    observed in student interaction sequences. This is a lightweight
    sequence model that serves as a stronger baseline than popularity.
    """

    def __init__(self):
        self.transition_counts: Dict[str, Counter] = defaultdict(Counter)
        self.topic_popularity: Counter = Counter()
        self.student_last_topic: Dict[int, str] = {}

    def fit(self, interactions: pd.DataFrame, order_col: Optional[str] = None) -> None:
        """
        Fit sequential transition model.

        Args:
            interactions: DataFrame with columns [student_id, topic, ...]
            order_col: Optional name of a timestamp/order column for sequences
        """
        data = interactions.copy()
        if order_col and order_col in data.columns:
            data = data.sort_values(order_col)
        elif 'timestamp' in data.columns:
            data = data.sort_values('timestamp')

        for student_id, df_student in data.groupby('student_id', sort=False):
            topics = df_student['topic'].tolist()
            if not topics:
                continue

            self.topic_popularity.update(topics)
            self.student_last_topic[int(student_id)] = topics[-1]

            for prev_topic, next_topic in zip(topics, topics[1:]):
                self.transition_counts[prev_topic][next_topic] += 1

    def recommend(self, student_id: int, topics: List[str], top_k: int = 5) -> List[Dict]:
        """Recommend top-k topics for a student based on transition counts."""
        if not topics:
            return []

        last_topic = self.student_last_topic.get(student_id)
        scores = []

        if last_topic and last_topic in self.transition_counts:
            next_counts = self.transition_counts[last_topic]
            total = sum(next_counts.values())
            if total > 0:
                for topic in topics:
                    score = float(next_counts.get(topic, 0)) / float(total)
                    scores.append((topic, score))

        if not scores or all(score <= 0 for _, score in scores):
            total = sum(self.topic_popularity.values()) or 1
            scores = [
                (topic, float(self.topic_popularity.get(topic, 0)) / float(total))
                for topic in topics
            ]

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"topic": topic, "score": score} for topic, score in scores[:top_k]]

    def explain_v2(self, student_id: int, interactions_df, mastery, kg, student_embeddings=None, topic_embeddings=None, top_k: int = 5, **kwargs):
        topics = sorted(interactions_df['topic'].dropna().unique().tolist())
        recs = self.recommend(student_id, topics, top_k=top_k)
        return [{"topic": rec['topic'], "recommendation_score": rec['score']} for rec in recs]


class HybridRecommender:
    """
    Hybrid Recommender combining CF and Content-Based
    """
    
    def __init__(self, cf_weight: float = 0.5, content_weight: float = 0.5):
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.cf_model = CollaborativeFilteringBaseline()
        self.content_model = ContentBasedFilteringBaseline()
    
    def fit(self, interactions: pd.DataFrame, topic_embeddings: Dict[str, np.ndarray]) -> None:
        """Fit both models"""
        self.cf_model.fit(interactions)
        self.content_model.fit(interactions, topic_embeddings)
    
    def recommend(self, student_id: int, topics: List[str], top_k: int = 5) -> List[Dict]:
        """Get hybrid recommendations"""
        cf_recs = self.cf_model.recommend(student_id, topics, top_k=len(topics))
        content_recs = self.content_model.recommend(student_id, topics, top_k=len(topics))
        
        # Merge scores
        cf_scores = {r['topic']: r['score'] for r in cf_recs}
        content_scores = {r['topic']: r['score'] for r in content_recs}
        
        hybrid_scores = []
        for topic in topics:
            cf_score = cf_scores.get(topic, 0.0)
            content_score = content_scores.get(topic, 0.0)
            hybrid_score = self.cf_weight * cf_score + self.content_weight * content_score
            hybrid_scores.append((topic, hybrid_score))
        
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"topic": topic, "score": score}
            for topic, score in hybrid_scores[:top_k]
        ]


def create_all_baselines(interactions: pd.DataFrame, 
                         topic_embeddings: Dict[str, np.ndarray]) -> Dict:
    """Create and fit all baseline models"""
    logger.info("Creating and training baseline recommenders...")
    
    baselines = {}
    
    # Popularity baseline
    pop = PopularityBaseline()
    pop.fit(interactions)
    baselines['popularity'] = pop

    # Collaborative Filtering
    cf = CollaborativeFilteringBaseline(n_factors=20)
    cf.fit(interactions)
    baselines['collaborative_filtering'] = cf
    
    # Content-Based
    cb = ContentBasedFilteringBaseline()
    cb.fit(interactions, topic_embeddings)
    baselines['content_based'] = cb
    
    # Sequential transition baseline
    sequential = SequentialTransitionBaseline()
    sequential.fit(interactions)
    baselines['sequential'] = sequential

    # Hybrid
    hybrid = HybridRecommender()
    hybrid.fit(interactions, topic_embeddings)
    baselines['hybrid'] = hybrid
    
    logger.info("✓ All baseline models trained")
    return baselines


def explain_recommendation(student_id: int,
                           topic: str,
                           embeddings: Dict[int, List[float]],
                           interactions: pd.DataFrame,
                           n_neighbors: int = 5,
                           features: Optional[Dict[int, Dict]] = None) -> Dict:
    """Explain why a topic is recommended for a student using nearest-student examples.

    Returns a dict with nearest neighbor ids, distances, neighbor_scores (avg on topic),
    and a short textual rationale.
    """
    import numpy as np

    ids = list(embeddings.keys())
    if student_id not in ids:
        # If target student not in embeddings, return baseline explanation
        global_avg = interactions['score'].mean() if 'score' in interactions.columns else 0.5
        return {
            'student_id': student_id,
            'topic': topic,
            'reason': 'No embedding for student; using global averages',
            'global_avg_score': float(global_avg),
            'neighbors': []
        }

    X = np.array([embeddings[int(i)] for i in ids])
    idx_map = {int(sid): i for i, sid in enumerate(ids)}
    target_vec = X[idx_map[student_id]]

    # Compute distances
    dists = np.linalg.norm(X - target_vec, axis=1)
    pairs = list(zip([int(i) for i in ids], dists))
    pairs = [p for p in pairs if p[0] != student_id]
    pairs.sort(key=lambda x: x[1])
    selected = pairs[:n_neighbors]

    neighbors_info = []
    for nid, dist in selected:
        # compute neighbor's avg score on topic
        nb_df = interactions[interactions['student_id'] == nid]
        if topic in nb_df['topic'].values:
            topic_scores = nb_df[nb_df['topic'] == topic]['score'].values
            avg_score = float(topic_scores.mean())
            success_rate = float((topic_scores > 0.7).mean())
        else:
            avg_score = float(nb_df['score'].mean()) if len(nb_df) > 0 else 0.0
            success_rate = float((nb_df['score'] > 0.7).mean()) if len(nb_df) > 0 else 0.0

        neighbors_info.append({
            'student_id': int(nid),
            'distance': float(dist),
            'avg_score_on_topic': avg_score,
            'success_rate_on_topic': success_rate
        })

    # Aggregate neighbor evidence
    avg_neighbor_score = float(np.mean([n['avg_score_on_topic'] for n in neighbors_info])) if neighbors_info else 0.0
    prop_success = float(np.mean([n['success_rate_on_topic'] for n in neighbors_info])) if neighbors_info else 0.0

    reason = (
        f"Nearest {len(neighbors_info)} similar students have average score {avg_neighbor_score:.2f} "
        f"and {prop_success*100:.0f}% success rate on this topic."
    )

    result: Dict = {
        'student_id': int(student_id),
        'topic': topic,
        'reason': reason,
        'neighbor_evidence': neighbors_info,
        'avg_neighbor_score': avg_neighbor_score,
        'prop_success': prop_success
    }

    # If features provided, compute per-feature deltas between target and neighbors
    if features:
        target_features = features.get(int(student_id))
        if target_features:
            # compute average neighbor features
            import math
            numeric_keys = [k for k, v in target_features.items() if isinstance(v, (int, float))]
            neighbor_avgs = {k: 0.0 for k in numeric_keys}
            count = 0
            for nb in neighbors_info:
                nbf = features.get(int(nb['student_id']))
                if not nbf:
                    continue
                count += 1
                for k in numeric_keys:
                    neighbor_avgs[k] += float(nbf.get(k, 0.0))

            if count > 0:
                for k in numeric_keys:
                    neighbor_avgs[k] /= float(count)

                deltas = {k: float(target_features[k]) - float(neighbor_avgs[k]) for k in numeric_keys}
                # rank features by absolute delta
                ranked = sorted(deltas.items(), key=lambda x: -abs(x[1]))
                # top 3 contributing features
                top = ranked[:3]
                # human-readable descriptions
                descriptions = [f"{k}: target {target_features[k]:.2f} vs neighbors {neighbor_avgs[k]:.2f} (delta {d:.2f})" for k, d in top]

                result['feature_deltas'] = {k: v for k, v in deltas.items()}
                result['top_feature_deltas'] = top
                result['feature_rationale'] = descriptions

    return result
