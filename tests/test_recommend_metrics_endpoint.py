import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from backend.main import app


def test_metrics_endpoint_full_and_temporal():
    with TestClient(app) as client:
        # Full evaluation
        resp = client.get('/recommend/metrics?top_k=3&evaluation=full')
        assert resp.status_code == 200
        payload = resp.json()
        assert 'metrics' in payload
        # keys exist and numeric
        metrics = payload['metrics']
        assert any(k.startswith('precision_at_') for k in metrics.keys())

        # Temporal evaluation
        resp2 = client.get('/recommend/metrics?top_k=3&evaluation=temporal&sample_fraction=0.2')
        assert resp2.status_code == 200
        payload2 = resp2.json()
        assert 'metrics' in payload2
        metrics2 = payload2['metrics']
        assert 'mrr' in metrics2
        # values should be between 0 and 1
        for v in metrics2.values():
            assert isinstance(v, (int, float))
            assert v >= 0.0 and v <= 1.0


def test_benchmark_endpoint():
    with TestClient(app) as client:
        resp = client.get('/recommend/benchmark?top_k=5&evaluation=full&sample_fraction=0.2')
        assert resp.status_code == 200
        payload = resp.json()
        assert 'popularity' in payload
        assert 'collaborative_filtering' in payload
        assert 'sequential' in payload
        assert 'explainable' in payload
        for metrics in [payload['popularity'], payload['collaborative_filtering'], payload['sequential'], payload['explainable']]:
            assert 'precision_at_5' in metrics
            assert 'recall_at_5' in metrics
            assert 'ndcg_at_5' in metrics
            assert 'mrr' in metrics
            for v in metrics.values():
                assert isinstance(v, (int, float))
                assert 0.0 <= v <= 1.0
