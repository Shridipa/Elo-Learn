import sys
import pathlib
import numpy as np
import pandas as pd

# Ensure project root on path so imports work under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from recommendation_engine.baselines import explain_recommendation


def test_explain_recommendation_feature_deltas():
    # synthetic embeddings for 4 students (ids 1-4)
    embeddings = {
        1: [0.1, 0.2, 0.3],
        2: [0.11, 0.19, 0.31],
        3: [0.5, 0.4, 0.3],
        4: [0.12, 0.22, 0.28]
    }

    # interactions: student 1 has low scores on topic 'A', neighbors have higher
    rows = []
    for sid in [1,2,3,4]:
        for i in range(3):
            rows.append({'student_id': sid, 'topic': 'A', 'score': 0.2 if sid==1 else 0.8})
    interactions = pd.DataFrame(rows)

    # features mapping with a clear delta: student 1 has lower success_rate
    features = {
        1: {'success_rate': 0.2, 'mean_score': 0.2, 'engagement': 0.1},
        2: {'success_rate': 0.8, 'mean_score': 0.8, 'engagement': 0.5},
        3: {'success_rate': 0.8, 'mean_score': 0.8, 'engagement': 0.6},
        4: {'success_rate': 0.75, 'mean_score': 0.75, 'engagement': 0.4},
    }

    explanation = explain_recommendation(1, 'A', embeddings, interactions, n_neighbors=3, features=features)
    assert 'feature_deltas' in explanation
    assert 'top_feature_deltas' in explanation
    assert 'feature_rationale' in explanation
    # Check that success_rate delta is negative (target lower than neighbors)
    assert explanation['feature_deltas']['success_rate'] < 0


if __name__ == '__main__':
    test_explain_recommendation_feature_deltas()
    print('OK')
