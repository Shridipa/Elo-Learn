"""Evaluation utilities for knowledge tracing."""
from typing import Dict, Any, List

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
except ImportError:
    accuracy_score = precision_score = recall_score = roc_auc_score = None


def compute_metrics(predictions: List[float], labels: List[int]) -> Dict[str, Any]:
    if len(predictions) == 0 or len(labels) == 0 or len(predictions) != len(labels):
        return {
            'auc': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'count': 0
        }

    binary_preds = [1 if p >= 0.5 else 0 for p in predictions]
    metrics = {'count': len(predictions)}

    try:
        if accuracy_score is not None:
            metrics['accuracy'] = float(accuracy_score(labels, binary_preds))
        else:
            metrics['accuracy'] = float(sum(1 for p, y in zip(binary_preds, labels) if p == y) / len(labels))
    except Exception:
        metrics['accuracy'] = 0.0

    try:
        if precision_score is not None:
            metrics['precision'] = float(precision_score(labels, binary_preds, zero_division=0))
        else:
            true_positive = sum(1 for p, y in zip(binary_preds, labels) if p == 1 and y == 1)
            predicted_positive = sum(binary_preds)
            metrics['precision'] = float(true_positive / predicted_positive) if predicted_positive > 0 else 0.0
    except Exception:
        metrics['precision'] = 0.0

    try:
        if recall_score is not None:
            metrics['recall'] = float(recall_score(labels, binary_preds, zero_division=0))
        else:
            true_positive = sum(1 for p, y in zip(binary_preds, labels) if p == 1 and y == 1)
            actual_positive = sum(labels)
            metrics['recall'] = float(true_positive / actual_positive) if actual_positive > 0 else 0.0
    except Exception:
        metrics['recall'] = 0.0

    try:
        if roc_auc_score is not None and len(set(labels)) > 1:
            metrics['auc'] = float(roc_auc_score(labels, predictions))
        else:
            metrics['auc'] = float(metrics['accuracy'])
    except Exception:
        metrics['auc'] = 0.0

    return metrics
