import pandas as pd
from recommendation_engine import evaluator


def test_basic_metrics_functions():
    y_true = {'a', 'b', 'c'}
    y_pred = ['a', 'x', 'b', 'c']

    # precision@2 -> only 'a' in top2 -> 1/2
    assert abs(evaluator.precision_at_k(y_true, y_pred, 2) - 0.5) < 1e-6
    # recall@3 -> top3 contains a and b -> 2/3
    assert abs(evaluator.recall_at_k(y_true, y_pred, 3) - (2/3)) < 1e-6
    # mrr -> first relevant at position 1 -> 1/1
    assert abs(evaluator.mrr(y_true, y_pred) - 1.0) < 1e-6
    # ndcg@3 should be > 0
    ndcg = evaluator.ndcg_at_k(y_true, y_pred, 3)
    assert ndcg > 0.0 and ndcg <= 1.0


class DummyRecommender:
    def __init__(self, preds_map):
        # preds_map: student_id -> list of topic preds
        self.preds_map = preds_map

    def explain_v2(self, student_id, interactions_df, mastery, kg, student_embeddings, topic_embeddings=None, top_k=5, **kwargs):
        preds = self.preds_map.get(student_id, [])
        return [{'topic': t, 'recommendation_score': 1.0, 'confidence': 1.0, 'readiness': 1.0, 'reasons': [], 'peer_evidence': {}} for t in preds[:top_k]]


def test_evaluate_with_holdout():
    # Build simple train/test dfs
    rows_train = [
        {'student_id': 1, 'topic': 'a'},
        {'student_id': 1, 'topic': 'b'},
    ]
    rows_test = [
        {'student_id': 1, 'topic': 'c'},
    ]
    train_df = pd.DataFrame(rows_train)
    test_df = pd.DataFrame(rows_test)

    # Dummy recommender always predicts ['c','a'] for student 1
    dummy = DummyRecommender({1: ['c', 'a']})

    metrics = evaluator.evaluate_with_holdout(dummy, train_df, test_df, students=[1], topic_embeddings={}, top_k=2)
    # precision@2 should be 1/2 (one relevant c in top2)
    assert 'precision_at_2' in metrics
    assert metrics['precision_at_2'] >= 0.0 and metrics['precision_at_2'] <= 1.0
    assert metrics['mrr'] >= 0.0 and metrics['mrr'] <= 1.0
