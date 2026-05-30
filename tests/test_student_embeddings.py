import sys
import pathlib
import pytest
import pandas as pd
from pathlib import Path

# Ensure project root on path so imports work under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ml_models.student_embeddings import StudentEmbeddingEngine


def test_aggregate_and_embedding_runs():
    csv_path = Path('datasets/student_interactions.csv')
    assert csv_path.exists(), 'dataset CSV must exist for test'
    df = pd.read_csv(csv_path)
    # Use subset to speed up test
    df_small = df[df['student_id'] <= 50]

    engine = StudentEmbeddingEngine(embedding_dim=16, results_path=Path('datasets'))
    embeddings, metadata = engine.fit_transform(df_small)

    # Basic assertions
    assert isinstance(embeddings, dict)
    assert len(embeddings) > 0
    sample_id = next(iter(embeddings.keys()))
    neighbors = engine.nearest_neighbors(embeddings, sample_id, n=3)
    assert isinstance(neighbors, list)
    # Allow for smaller datasets: expect up to 3 neighbors excluding the target
    expected = min(3, max(0, len(embeddings) - 1))
    assert len(neighbors) == expected


if __name__ == '__main__':
    pytest.main(['-q'])
