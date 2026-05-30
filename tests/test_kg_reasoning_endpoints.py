import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from backend.main import app


def _create_client():
    return TestClient(app)


def test_kg_prerequisites_endpoint():
    with _create_client() as client:
        response = client.get('/kg/prerequisites/Transformer Architecture')
    assert response.status_code == 200
    payload = response.json()
    assert payload['topic'] == 'Transformer Architecture'
    assert 'Backpropagation' in payload['prerequisites']
    assert 'Attention Mechanisms' in payload['prerequisites']


def test_kg_path_endpoint():
    with _create_client() as client:
        response = client.get('/kg/path?topic=Transformer Architecture')
        assert response.status_code == 200
    payload = response.json()
    assert payload['topic'] == 'Transformer Architecture'
    assert isinstance(payload['path'], list)
    assert payload['path'][-1] == 'Transformer Architecture'
    assert 'Backpropagation' in payload['path']
    assert 'Attention Mechanisms' in payload['path']


def test_kg_readiness_endpoint_for_student():
    with _create_client() as client:
        response = client.get('/kg/readiness/1?topic=Transformer Architecture')
        assert response.status_code == 200
    payload = response.json()
    assert payload['student_id'] == 1
    assert 'readiness' in payload
    assert 0.0 <= payload['readiness'] <= 1.0


def test_kg_root_cause_endpoint_for_student():
    with _create_client() as client:
        response = client.get('/kg/root_cause/1?topic=Transformer Architecture')
        assert response.status_code == 200
    payload = response.json()
    assert payload['student_id'] == 1
    assert 'root_cause' in payload
    root_cause = payload['root_cause']
    assert root_cause['target'] == 'Transformer Architecture'
    assert isinstance(root_cause['weak_prerequisites'], list)


def test_kg_remediation_endpoint_for_student():
    with _create_client() as client:
        response = client.get('/kg/remediation/1?topic=Transformer Architecture')
        assert response.status_code == 200
        payload = response.json()
        assert payload['student_id'] == 1
        assert 'remediation' in payload
        remediation = payload['remediation']
        assert remediation['target'] == 'Transformer Architecture'
        assert isinstance(remediation['remediation_plan'], list)
        assert isinstance(remediation['full_path'], list)
