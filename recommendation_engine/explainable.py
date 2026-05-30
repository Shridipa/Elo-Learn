"""Explainable Recommendation Engine (V2)

Combines student embeddings, knowledge tracing mastery, knowledge graph
readiness, and cluster/peer evidence to produce evidence-backed
recommendations with explanations and confidence scores.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class ExplainableRecommender:
    def __init__(self, neighbor_count: int = 10):
        self.neighbor_count = neighbor_count

    def _cosine(self, a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _nearest_neighbors(self, student_id: int, embeddings: Dict[int, List[float]], top_k: int):
        ids = list(embeddings.keys())
        if student_id not in ids:
            return []
        X = np.array([embeddings[int(i)] for i in ids])
        idx_map = {int(sid): i for i, sid in enumerate(ids)}
        target = X[idx_map[student_id]]
        dists = np.linalg.norm(X - target, axis=1)
        pairs = list(zip([int(i) for i in ids], dists))
        pairs = [p for p in pairs if p[0] != student_id]
        pairs.sort(key=lambda x: x[1])
        return pairs[:top_k]

    def explain_v2(self,
                   student_id: int,
                   interactions_df,
                   mastery: Dict[str, float],
                   kg,
                   student_embeddings: Dict[int, List[float]],
                   topic_embeddings: Optional[Dict[str, Any]] = None,
                   top_k: int = 5,
                   weights: Optional[Dict[str, float]] = None,
                   cluster_labels: Optional[Dict[int, int]] = None,
                   ) -> List[Dict[str, Any]]:
        """Generate top-k explainable recommendations.

        Returns a list of recommendation dicts containing score, confidence,
        readiness, reasons and peer evidence.
        """
        if weights is None:
            weights = {'similarity': 0.35, 'mastery': 0.25, 'readiness': 0.25, 'cluster_success': 0.15}

        # Determine candidate topics
        try:
            topics = sorted(interactions_df['topic'].dropna().unique().tolist())
        except Exception:
            topics = list(kg.graph.nodes())

        # student embedding
        stud_emb = None
        if int(student_id) in student_embeddings:
            stud_emb = student_embeddings[int(student_id)]

        recs = []
        for topic in topics:
            # Similarity between student and topic embedding
            sim = 0.0
            if stud_emb is not None and topic_embeddings and topic in topic_embeddings:
                sim = self._cosine(stud_emb, topic_embeddings[topic])
                # normalize cosine from [-1,1] to [0,1]
                sim = float((sim + 1.0) / 2.0)

            # Mastery score (direct concept mastery if available)
            mastery_score = float(mastery.get(topic, 0.0))

            # Readiness from knowledge graph
            try:
                readiness = float(kg.get_readiness(topic, mastery))
            except Exception:
                readiness = 0.0

            # Cluster success: average success rate for topic within student's cluster
            cluster_success = 0.0
            if cluster_labels and int(student_id) in cluster_labels:
                cluster = cluster_labels.get(int(student_id))
                members = [sid for sid, lbl in cluster_labels.items() if lbl == cluster]
                if members:
                    sub = interactions_df[interactions_df['student_id'].isin(members)]
                    if topic in sub['topic'].values:
                        topic_scores = sub[sub['topic'] == topic]['score'].values
                        if len(topic_scores) > 0:
                            cluster_success = float((topic_scores > 0.7).mean())

            # Peer neighbors evidence
            neighbors = self._nearest_neighbors(student_id, student_embeddings, top_k=self.neighbor_count) if student_embeddings else []
            neighbor_ids = [n[0] for n in neighbors]
            neighbor_count = len(neighbor_ids)
            peer_success = 0.0
            if neighbor_count > 0:
                nb_df = interactions_df[interactions_df['student_id'].isin(neighbor_ids)]
                if topic in nb_df['topic'].values:
                    topic_scores = nb_df[nb_df['topic'] == topic]['score'].values
                    if len(topic_scores) > 0:
                        peer_success = float((topic_scores > 0.7).mean())

            # Compose final recommendation score using configured weights
            final_score = (
                weights.get('similarity', 0.0) * sim
                + weights.get('mastery', 0.0) * mastery_score
                + weights.get('readiness', 0.0) * readiness
                + weights.get('cluster_success', 0.0) * cluster_success
            )

            # Confidence heuristic: emphasize readiness + peer evidence
            confidence = 0.5 * readiness + 0.3 * sim + 0.2 * cluster_success

            # Reasons (human-readable)
            reasons = []
            if mastery_score >= 0.8:
                reasons.append('Strong concept mastery')
            elif mastery_score >= 0.6:
                reasons.append('Moderate concept mastery')
            else:
                reasons.append('Low concept mastery')

            if readiness >= 0.8:
                reasons.append('Readiness above threshold')
            elif readiness >= 0.5:
                reasons.append('Partially ready; review prerequisites')
            else:
                reasons.append('Prerequisites incomplete')

            # prerequisites info
            missing = kg.get_missing_prerequisites(topic, mastery)
            if missing:
                reasons.append(f"Missing prerequisites: {', '.join(missing[:3])}")
            else:
                reasons.append('Completed prerequisite chain')

            if sim >= 0.6:
                reasons.append('High embedding similarity to topic')

            # Peer evidence summary
            peer_evidence = {
                'similar_students': neighbor_count,
                'success_rate': round(peer_success, 3)
            }

            recs.append({
                'topic': topic,
                'recommendation_score': round(float(final_score), 4),
                'confidence': round(float(confidence), 4),
                'readiness': round(float(readiness), 4),
                'reasons': reasons,
                'peer_evidence': peer_evidence,
            })

        # sort and return top_k
        recs.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return recs[:top_k]

    def evidence_summary(self, student_id: int, interactions_df, student_embeddings: Dict[int, List[float]], top_k: int = 5):
        """Aggregate peer evidence for a student across topics."""
        neighbors = self._nearest_neighbors(student_id, student_embeddings, top_k)
        neighbor_ids = [n[0] for n in neighbors]
        neighbor_count = len(neighbor_ids)
        if neighbor_count == 0:
            return {'similar_students': 0, 'average_mastery': 0.0, 'completion_rate': 0.0, 'success_rate': 0.0}

        # average mastery per neighbor (approx using interactions)
        avg_scores = []
        completions = []
        successes = []
        for nid in neighbor_ids:
            nb = interactions_df[interactions_df['student_id'] == nid]
            if nb.empty:
                continue
            avg_scores.append(float(nb['score'].mean()))
            completions.append(int(nb['topic'].notnull().sum() > 0))
            successes.append(float((nb['score'] > 0.7).mean()))

        return {
            'similar_students': neighbor_count,
            'neighbor_ids': neighbor_ids,
            'average_mastery': round(float(np.mean(avg_scores)) if avg_scores else 0.0, 3),
            'completion_rate': round(float(np.mean(completions)) if completions else 0.0, 3),
            'success_rate': round(float(np.mean(successes)) if successes else 0.0, 3)
        }

    def confidence_summary(self, student_id: int, interactions_df, mastery: Dict[str, float], kg, student_embeddings: Dict[int, List[float]], topic_embeddings: Optional[Dict[str, Any]] = None, top_k: int = 5):
        """Return average confidence across top_k recommended topics and per-topic summary."""
        recs = self.explain_v2(student_id, interactions_df, mastery, kg, student_embeddings, topic_embeddings=topic_embeddings, top_k=top_k)
        avg_conf = float(np.mean([r['confidence'] for r in recs])) if recs else 0.0
        return {'average_confidence': round(avg_conf, 4), 'recommendations': recs}

    def readiness_summary(self, student_id: int, mastery: Dict[str, float], kg, topic: Optional[str] = None):
        if topic:
            readiness = kg.get_readiness(topic, mastery)
            return {'readiness': float(readiness)}
        scores = kg.get_readiness_scores(mastery)
        return {'readiness': scores}
"""Explainable recommendation utilities for Elo Learn."""

import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from knowledge_graph.graph import KnowledgeGraph

DEFAULT_RECOMMENDATION_WEIGHTS = {
    'similarity': 0.35,
    'mastery': 0.25,
    'readiness': 0.25,
    'cluster_success': 0.15,
}

class ExplainableRecommender:
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 mastery_threshold: float = 0.7,
                 neighbor_count: int = 10):
        self.weights = weights or DEFAULT_RECOMMENDATION_WEIGHTS.copy()
        self.mastery_threshold = mastery_threshold
        self.neighbor_count = neighbor_count

    @staticmethod
    def _normalize_score(value: float) -> float:
        return float(min(max(value, 0.0), 1.0))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        # Handle dimension mismatch by using minimum dimensions
        min_dim = min(len(a), len(b))
        a_aligned = a[:min_dim]
        b_aligned = b[:min_dim]
        norm_a = np.linalg.norm(a_aligned)
        norm_b = np.linalg.norm(b_aligned)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_aligned, b_aligned) / (norm_a * norm_b))

    def _vector_for(self, obj: Any) -> Optional[np.ndarray]:
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, list):
            return np.array(obj, dtype=np.float32)
        return None

    def _nearest_neighbors(self,
                           student_id: int,
                           embeddings: Dict[int, Any],
                           n_neighbors: Optional[int] = None) -> List[int]:
        if n_neighbors is None:
            n_neighbors = self.neighbor_count
        ids = sorted([int(k) for k in embeddings.keys()])
        if student_id not in ids:
            return []
        X = np.array([self._vector_for(embeddings[int(i)]) for i in ids])
        target_idx = ids.index(student_id)
        target_vec = X[target_idx]
        if target_vec is None or np.linalg.norm(target_vec) == 0:
            return []

        distances = np.linalg.norm(X - target_vec, axis=1)
        pairs = [(int(sid), float(dist)) for sid, dist in zip(ids, distances) if int(sid) != student_id]
        pairs.sort(key=lambda x: x[1])
        return [sid for sid, _ in pairs[:n_neighbors]]

    def _topic_similarity(self,
                          student_id: int,
                          student_embeddings: Dict[int, Any],
                          topic: str,
                          topic_embeddings: Dict[str, Any]) -> float:
        student_vector = self._vector_for(student_embeddings.get(int(student_id)))
        topic_vector = self._vector_for(topic_embeddings.get(topic))
        similarity = self._cosine_similarity(student_vector, topic_vector)
        return self._normalize_score((similarity + 1.0) / 2.0)

    def _mastery_score(self,
                       topic: str,
                       mastery_levels: Dict[str, float],
                       kg: KnowledgeGraph) -> float:
        if topic in mastery_levels:
            return self._normalize_score(float(mastery_levels.get(topic, 0.0)))
        prerequisites = kg.get_prerequisites(topic)
        if not prerequisites:
            return 0.0
        values = [mastery_levels.get(prereq, 0.0) for prereq in prerequisites]
        return self._normalize_score(float(np.mean(values)))

    def _cluster_success_score(self,
                               student_id: int,
                               topic: str,
                               embeddings: Dict[int, Any],
                               interactions: pd.DataFrame,
                               cluster_labels: Optional[Dict[int, int]] = None) -> float:
        neighbor_ids = self._nearest_neighbors(student_id, embeddings, self.neighbor_count)
        topic_success = []

        if len(neighbor_ids) > 0:
            for nid in neighbor_ids:
                neighbor_df = interactions[interactions['student_id'] == nid]
                if topic in neighbor_df['topic'].values:
                    scores = neighbor_df[neighbor_df['topic'] == topic]['score'].astype(float).tolist()
                    if scores:
                        topic_success.append(float(np.mean(scores)))
                else:
                    if len(neighbor_df) > 0:
                        topic_success.append(float(neighbor_df['score'].astype(float).mean()))

        if cluster_labels and int(student_id) in cluster_labels:
            cluster_id = cluster_labels[int(student_id)]
            cluster_peer_ids = [sid for sid, lbl in cluster_labels.items() if lbl == cluster_id and sid != int(student_id)]
            for cid in cluster_peer_ids:
                cluster_df = interactions[interactions['student_id'] == cid]
                if topic in cluster_df['topic'].values:
                    scores = cluster_df[cluster_df['topic'] == topic]['score'].astype(float).tolist()
                    if scores:
                        topic_success.append(float(np.mean(scores)))

        if not topic_success:
            return 0.0
        return self._normalize_score(float(np.mean(topic_success)))

    def _compute_peer_evidence(self,
                               student_id: int,
                               topic: str,
                               embeddings: Dict[int, Any],
                               interactions: pd.DataFrame) -> Dict[str, Any]:
        neighbor_ids = self._nearest_neighbors(student_id, embeddings, self.neighbor_count)
        evidence = {
            'similar_students': len(neighbor_ids),
            'success_rate': 0.0,
            'average_mastery': 0.0,
            'neighbor_ids': neighbor_ids,
            'topic': topic,
        }

        if not neighbor_ids:
            return evidence

        scores = []
        successes = []
        for nid in neighbor_ids:
            neighbor_df = interactions[interactions['student_id'] == nid]
            if topic in neighbor_df['topic'].values:
                topic_scores = neighbor_df[neighbor_df['topic'] == topic]['score'].astype(float).tolist()
                if topic_scores:
                    scores.append(float(np.mean(topic_scores)))
                    successes.append(float(np.mean([1.0 if s >= 0.7 else 0.0 for s in topic_scores])))
            elif len(neighbor_df) > 0:
                scores.append(float(neighbor_df['score'].astype(float).mean()))
                successes.append(float(np.mean([1.0 if s >= 0.7 else 0.0 for s in neighbor_df['score'].astype(float).tolist()])))

        evidence['average_mastery'] = float(np.mean(scores)) if scores else 0.0
        evidence['success_rate'] = float(np.mean(successes)) if successes else 0.0
        return evidence

    def _reasons(self,
                 topic: str,
                 mastery_score: float,
                 similarity_score: float,
                 readiness_score: float,
                 cluster_success_score: float,
                 kg: KnowledgeGraph,
                 mastery_levels: Dict[str, float]) -> List[str]:
        reasons = []
        if similarity_score >= 0.65:
            reasons.append('Embedding similarity strongly supports this topic')
        elif similarity_score >= 0.4:
            reasons.append('Moderate embedding alignment with topic concepts')

        if mastery_score >= 0.75:
            reasons.append('Strong existing mastery on this topic or its prerequisites')
        elif mastery_score >= 0.5:
            reasons.append('Partial mastery of prerequisite concepts')
        else:
            reasons.append('This topic offers valuable remediation opportunities')

        missing_prereqs = kg.get_missing_prerequisites(topic, mastery_levels, mastery_threshold=self.mastery_threshold)
        if not missing_prereqs:
            reasons.append('Prerequisite chain is complete or sufficiently practiced')
        else:
            reasons.append(f"Bridging {len(missing_prereqs)} prerequisite gap(s): {', '.join(missing_prereqs[:3])}")

        if readiness_score >= 0.7:
            reasons.append('Readiness is high based on knowledge tracing and prerequisite mastery')
        elif readiness_score >= 0.5:
            reasons.append('Readiness is moderate and worth continuing with guided support')
        else:
            reasons.append('Readiness is low; focus on remediation before advanced topics')

        if cluster_success_score >= 0.6:
            reasons.append('Peers in the same learning cohort show strong success on this topic')
        elif cluster_success_score >= 0.35:
            reasons.append('Cluster peers show moderate progress on this topic')

        return reasons[:5]

    def _confidence(self,
                    similarity_score: float,
                    readiness_score: float,
                    cluster_success_score: float,
                    mastery_score: float) -> float:
        confidence = (
            0.35 * readiness_score +
            0.25 * cluster_success_score +
            0.25 * similarity_score +
            0.15 * mastery_score
        )
        return self._normalize_score(confidence)

    def _topic_embeddings(self,
                          kg: KnowledgeGraph,
                          external_topic_embeddings: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        if external_topic_embeddings:
            return {topic: self._vector_for(emb) for topic, emb in external_topic_embeddings.items() if self._vector_for(emb) is not None}
        return kg.compute_concept_embeddings()

    def explain_v2(self,
                   student_id: int,
                   interactions: pd.DataFrame,
                   mastery_levels: Dict[str, float],
                   kg: KnowledgeGraph,
                   student_embeddings: Dict[int, Any],
                   topic_embeddings: Optional[Dict[str, Any]] = None,
                   top_k: int = 5,
                   weights: Optional[Dict[str, float]] = None,
                   cluster_labels: Optional[Dict[int, int]] = None) -> List[Dict[str, Any]]:
        if weights is not None:
            self.weights = weights

        topics = sorted(interactions['topic'].dropna().unique().tolist())
        topic_embedding_map = self._topic_embeddings(kg, topic_embeddings)
        recommendations: List[Dict[str, Any]] = []

        for topic in topics:
            similarity_score = self._topic_similarity(student_id, student_embeddings, topic, topic_embedding_map)
            mastery_score = self._mastery_score(topic, mastery_levels, kg)
            readiness_score = kg.get_readiness(topic, mastery_levels, mastery_threshold=self.mastery_threshold)
            cluster_success_score = self._cluster_success_score(
                student_id, topic, student_embeddings, interactions,
                cluster_labels=cluster_labels
            )
            recommendation_score = self._normalize_score(
                self.weights['similarity'] * similarity_score +
                self.weights['mastery'] * mastery_score +
                self.weights['readiness'] * readiness_score +
                self.weights['cluster_success'] * cluster_success_score
            )
            confidence = self._confidence(similarity_score, readiness_score, cluster_success_score, mastery_score)
            peer_evidence = self._compute_peer_evidence(student_id, topic, student_embeddings, interactions)
            reasons = self._reasons(topic, mastery_score, similarity_score, readiness_score, cluster_success_score, kg, mastery_levels)

            recommendations.append({
                'topic': topic,
                'recommendation_score': recommendation_score,
                'confidence': confidence,
                'readiness': readiness_score,
                'mastery': mastery_score,
                'similarity_score': similarity_score,
                'cluster_success_score': cluster_success_score,
                'reasons': reasons,
                'peer_evidence': peer_evidence,
            })

        recommendations.sort(key=lambda item: item['recommendation_score'], reverse=True)
        return recommendations[:top_k]

    def evidence_summary(self,
                         student_id: int,
                         interactions: pd.DataFrame,
                         student_embeddings: Dict[int, Any],
                         top_k: int = 5) -> Dict[str, Any]:
        neighbor_ids = self._nearest_neighbors(student_id, student_embeddings, self.neighbor_count)
        evidence = {
            'student_id': student_id,
            'similar_students': len(neighbor_ids),
            'neighbor_ids': neighbor_ids,
            'average_mastery': 0.0,
            'average_success_rate': 0.0,
        }
        if not neighbor_ids:
            return evidence

        scores = []
        successes = []
        for nid in neighbor_ids:
            neighbor_df = interactions[interactions['student_id'] == nid]
            if len(neighbor_df) == 0:
                continue
            scores.append(float(neighbor_df['score'].astype(float).mean()))
            successes.append(float((neighbor_df['score'].astype(float) >= 0.7).mean()))

        evidence['average_mastery'] = float(np.mean(scores)) if scores else 0.0
        evidence['average_success_rate'] = float(np.mean(successes)) if successes else 0.0
        return evidence

    def confidence_summary(self,
                           student_id: int,
                           interactions: pd.DataFrame,
                           mastery_levels: Dict[str, float],
                           kg: KnowledgeGraph,
                           student_embeddings: Dict[int, Any],
                           topic_embeddings: Optional[Dict[str, Any]] = None,
                           top_k: int = 5) -> Dict[str, Any]:
        recommendations = self.explain_v2(
            student_id,
            interactions,
            mastery_levels,
            kg,
            student_embeddings,
            topic_embeddings=topic_embeddings,
            top_k=top_k
        )
        confidences = [rec['confidence'] for rec in recommendations] if recommendations else [0.0]
        return {
            'student_id': student_id,
            'average_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'recommendations': [{'topic': rec['topic'], 'confidence': rec['confidence']} for rec in recommendations]
        }

    def readiness_summary(self,
                          student_id: int,
                          mastery_levels: Dict[str, float],
                          kg: KnowledgeGraph,
                          topic: Optional[str] = None) -> Dict[str, Any]:
        if topic:
            return {
                'student_id': student_id,
                'topic': topic,
                'readiness': kg.get_readiness(topic, mastery_levels, self.mastery_threshold)
            }

        return {
            'student_id': student_id,
            'readiness': kg.get_readiness_scores(mastery_levels, self.mastery_threshold)
        }
