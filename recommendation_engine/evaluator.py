"""Recommendation evaluation metrics

Provides Precision@K, Recall@K, NDCG@K, MRR, Coverage, Diversity, Novelty
and a lightweight evaluation runner to compute aggregated metrics over a
set of users and recommendations.
"""
from typing import List, Set, Dict, Iterable, Optional
import numpy as np
import math


def precision_at_k(y_true: Set[str], y_pred: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    preds = y_pred[:k]
    if not preds:
        return 0.0
    hit = len(set(preds) & set(y_true))
    return hit / float(k)


def recall_at_k(y_true: Set[str], y_pred: List[str], k: int) -> float:
    if not y_true:
        return 0.0
    preds = y_pred[:k]
    hit = len(set(preds) & set(y_true))
    return hit / float(len(y_true))


def dcg_at_k(y_true: Set[str], y_pred: List[str], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(y_pred[:k]):
        rel = 1.0 if p in y_true else 0.0
        denom = math.log2(i + 2)
        dcg += (2 ** rel - 1) / denom
    return dcg


def idcg_at_k(y_true: Set[str], k: int) -> float:
    # ideal DCG when all relevant items ranked at top
    ideal_rels = min(len(y_true), k)
    idcg = 0.0
    for i in range(ideal_rels):
        idcg += (2 ** 1 - 1) / math.log2(i + 2)
    return idcg


def ndcg_at_k(y_true: Set[str], y_pred: List[str], k: int) -> float:
    idcg = idcg_at_k(y_true, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(y_true, y_pred, k) / idcg


def mrr(y_true: Set[str], y_pred: List[str]) -> float:
    for i, p in enumerate(y_pred):
        if p in y_true:
            return 1.0 / float(i + 1)
    return 0.0


def coverage(all_recommendations: Iterable[List[str]], all_items: Set[str]) -> float:
    recommended = set()
    for recs in all_recommendations:
        recommended.update(recs)
    if not all_items:
        return 0.0
    return len(recommended) / float(len(all_items))


def diversity(recommendations: Iterable[List[str]], topic_embeddings: Dict[str, List[float]]) -> float:
    # Average pairwise dissimilarity (1 - cosine similarity) across recommended lists
    def cosine(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    dissimilarities = []
    for recs in recommendations:
        vecs = [topic_embeddings[t] for t in recs if t in topic_embeddings]
        n = len(vecs)
        if n < 2:
            continue
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine(vecs[i], vecs[j])
                total += (1.0 - sim)
                count += 1
        if count > 0:
            dissimilarities.append(total / count)

    if not dissimilarities:
        return 0.0
    return float(np.mean(dissimilarities))


def novelty(recommendations: Iterable[List[str]], popularity: Dict[str, int]) -> float:
    # Popularity -> higher frequency means lower novelty. Use normalized inverse popularity
    if not popularity:
        return 0.0
    max_pop = max(popularity.values())
    if max_pop == 0:
        return 0.0

    novs = []
    for recs in recommendations:
        scores = []
        for t in recs:
            p = popularity.get(t, 0)
            norm = p / float(max_pop)
            scores.append(1.0 - norm)
        if scores:
            novs.append(float(np.mean(scores)))

    return float(np.mean(novs)) if novs else 0.0


def evaluate_all(recommender, interactions_df, students: Iterable[int],
                 topic_embeddings: Dict[str, List[float]] = None,
                 top_k: int = 5,
                 sample_fraction: float = 1.0) -> Dict[str, float]:
    """Evaluate recommender across provided students and return aggregated metrics."""
    if topic_embeddings is None:
        topic_embeddings = {}

    # Compute popularity
    topic_counts = interactions_df['topic'].value_counts().to_dict()

    all_items = set(interactions_df['topic'].dropna().unique().tolist())

    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    rec_lists = []

    students = list(students)
    if sample_fraction < 1.0:
        n = max(1, int(len(students) * sample_fraction))
        students = students[:n]

    for sid in students:
        # ground truth: topics the student has interacted with
        user_df = interactions_df[interactions_df['student_id'] == sid]
        y_true = set(user_df['topic'].dropna().unique().tolist())

        # predicted: use recommender.explain_v2 to generate top_k topics
        try:
            recs = recommender.explain_v2(sid, interactions_df, {}, None, {}, topic_embeddings=topic_embeddings, top_k=top_k)
            y_pred = [r['topic'] for r in recs]
        except Exception:
            y_pred = []

        rec_lists.append(y_pred)

        precisions.append(precision_at_k(y_true, y_pred, top_k))
        recalls.append(recall_at_k(y_true, y_pred, top_k))
        ndcgs.append(ndcg_at_k(y_true, y_pred, top_k))
        mrrs.append(mrr(y_true, y_pred))

    metrics = {
        f'precision_at_{top_k}': float(np.mean(precisions)) if precisions else 0.0,
        f'recall_at_{top_k}': float(np.mean(recalls)) if recalls else 0.0,
        f'ndcg_at_{top_k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
        'mrr': float(np.mean(mrrs)) if mrrs else 0.0,
        'coverage': coverage(rec_lists, all_items),
        'diversity': diversity(rec_lists, topic_embeddings),
        'novelty': novelty(rec_lists, topic_counts),
    }

    return metrics


def evaluate_with_holdout(recommender, train_df, test_df, students: Iterable[int],
                          topic_embeddings: Dict[str, List[float]] = None,
                          top_k: int = 5) -> Dict[str, float]:
    """Evaluate recommender using a provided train/test split.

    - `recommender` should be callable in the same way as used by evaluate_all:
      recommender.explain_v2(student_id, interactions_df=TRAIN_DF, ...)
    - `train_df` is used as the input to the recommender to generate predictions.
    - `test_df` is used to derive ground truth topics per student.
    """
    if topic_embeddings is None:
        topic_embeddings = {}

    all_items = set(test_df['topic'].dropna().unique().tolist()) if not test_df.empty else set()

    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    rec_lists = []

    for sid in students:
        # ground truth: topics in test_df for this student
        user_test = test_df[test_df['student_id'] == sid]
        y_true = set(user_test['topic'].dropna().unique().tolist())

        # skip users with no test items
        if not y_true:
            continue

        try:
            recs = recommender.explain_v2(sid, train_df, {}, None, {}, topic_embeddings=topic_embeddings, top_k=top_k)
            y_pred = [r['topic'] for r in recs]
        except Exception:
            y_pred = []

        rec_lists.append(y_pred)

        precisions.append(precision_at_k(y_true, y_pred, top_k))
        recalls.append(recall_at_k(y_true, y_pred, top_k))
        ndcgs.append(ndcg_at_k(y_true, y_pred, top_k))
        mrrs.append(mrr(y_true, y_pred))

    metrics = {
        f'precision_at_{top_k}': float(np.mean(precisions)) if precisions else 0.0,
        f'recall_at_{top_k}': float(np.mean(recalls)) if recalls else 0.0,
        f'ndcg_at_{top_k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
        'mrr': float(np.mean(mrrs)) if mrrs else 0.0,
        'coverage': coverage(rec_lists, all_items),
        'diversity': diversity(rec_lists, topic_embeddings),
        'novelty': novelty(rec_lists, train_df['topic'].value_counts().to_dict() if not train_df.empty else {}),
    }

    return metrics
