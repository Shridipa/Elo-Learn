import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from backend.main import app


def test_recommend_explain_v2_endpoint():
    with TestClient(app) as client:
        response = client.post('/recommend/explain_v2', json={'student_id': 1, 'top_k': 3})
        assert response.status_code == 200
        payload = response.json()
        assert payload['student_id'] == 1
        assert isinstance(payload['recommendations'], list)
        assert 'weights' in payload
        for rec in payload['recommendations']:
            assert 'topic' in rec
            assert 0.0 <= rec['recommendation_score'] <= 1.0
            assert 0.0 <= rec['confidence'] <= 1.0
            assert 0.0 <= rec['readiness'] <= 1.0
            assert isinstance(rec['reasons'], list)
            assert isinstance(rec['peer_evidence'], dict)


def test_recommend_evidence_endpoint():
    with TestClient(app) as client:
        response = client.get('/recommend/evidence/1?top_k=3')
        assert response.status_code == 200
        payload = response.json()
        assert payload['student_id'] == 1
        assert 'evidence' in payload
        assert 'recommendations' in payload
        assert isinstance(payload['evidence'].get('neighbor_ids', []), list)


def test_recommend_confidence_endpoint():
    with TestClient(app) as client:
        response = client.get('/recommend/confidence/1?top_k=2')
        assert response.status_code == 200
        payload = response.json()
        assert payload['student_id'] == 1
        assert 0.0 <= payload.get('average_confidence', 0.0) <= 1.0
        assert isinstance(payload.get('recommendations', []), list)


def test_recommend_readiness_endpoint():
    with TestClient(app) as client:
        response = client.get('/recommend/readiness/1')
        assert response.status_code == 200
        payload = response.json()
        assert payload['student_id'] == 1
        assert isinstance(payload['readiness'], dict)
