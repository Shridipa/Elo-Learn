import json
import tempfile
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

import sys
import pathlib as pl
sys.path.insert(0, str(pl.Path(__file__).resolve().parents[1]))

from backend.main import app
from backend.config import settings

client = TestClient(app)


def test_recommend_explain_endpoint(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None

    data_dir = settings.data_path
    data_dir.mkdir(parents=True, exist_ok=True)

    interactions = pd.DataFrame([
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.2},
        {'student_id': 2, 'topic': 'Matrices', 'score': 0.8},
        {'student_id': 3, 'topic': 'Matrices', 'score': 0.75},
    ])
    interactions_file = data_dir / 'student_interactions.csv'
    interactions.to_csv(interactions_file, index=False)

    # Create embeddings JSON with features
    embeddings = {
        'student_ids': [1,2,3],
        'embeddings': {
            '1': [0.0,0.0],
            '2': [0.1,0.1],
            '3': [0.2,0.1]
        },
        'features': {
            '1': {'mean_score': 0.2, 'success_rate': 0.0},
            '2': {'mean_score': 0.8, 'success_rate': 1.0},
            '3': {'mean_score': 0.75, 'success_rate': 1.0}
        }
    }
    emb_file = data_dir / 'student_embeddings.json'
    emb_file.write_text(json.dumps(embeddings))

    # Call the explain endpoint
    resp = client.post('/recommend/explain', json={'student_id': 1, 'topic': 'Matrices', 'n_neighbors': 2})
    assert resp.status_code == 200
    body = resp.json()
    assert 'explanation' in body
    exp = body['explanation']
    assert exp['student_id'] == 1
    assert 'neighbor_evidence' in exp
    assert 'feature_deltas' in exp


def test_cache_endpoints_refresh_and_status(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None

    embeddings = {
        'embeddings': {
            '1': [0.0, 0.0],
            '2': [0.1, 0.1],
            '3': [0.2, 0.1]
        },
        'features': {
            '1': {'mean_score': 0.2, 'success_rate': 0.0},
            '2': {'mean_score': 0.8, 'success_rate': 1.0},
            '3': {'mean_score': 0.75, 'success_rate': 1.0}
        }
    }
    emb_file = settings.data_path / 'student_embeddings.json'
    emb_file.write_text(json.dumps(embeddings))

    resp = client.post('/cache/refresh')
    assert resp.status_code == 200
    assert resp.json().get('cached') is True

    status_resp = client.get('/cache/status')
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body.get('cached') is True
    assert status_body.get('student_count') == 3


def test_student_profile_similarity_and_cluster_endpoints(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None

    interactions = pd.DataFrame([
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.2},
        {'student_id': 1, 'topic': 'Vectors', 'score': 0.8},
        {'student_id': 2, 'topic': 'Matrices', 'score': 0.75},
        {'student_id': 3, 'topic': 'Vectors', 'score': 0.9},
        {'student_id': 4, 'topic': 'Python Fundamentals', 'score': 0.6},
    ])
    interactions_file = settings.data_path / 'student_interactions.csv'
    interactions_file.write_text(interactions.to_csv(index=False))

    embeddings = {
        'student_ids': [1, 2, 3, 4],
        'embeddings': {
            '1': [0.0, 0.0, 0.1, 0.1],
            '2': [0.1, 0.0, 0.1, 0.2],
            '3': [0.0, 0.2, 0.2, 0.1],
            '4': [0.3, 0.1, 0.0, 0.0]
        },
        'cluster_labels': [0, 0, 1, 1],
        'coords_pca': {'1': [0.0, 0.0], '2': [0.1, 0.0], '3': [0.0, 0.2], '4': [0.3, 0.1]},
        'coords_tsne': {'1': [0.0, 0.0], '2': [0.1, -0.1], '3': [0.2, 0.2], '4': [0.3, 0.1]},
        'features': {
            '1': {'mean_score': 0.5, 'success_rate': 0.5},
            '2': {'mean_score': 0.75, 'success_rate': 1.0},
            '3': {'mean_score': 0.9, 'success_rate': 1.0},
            '4': {'mean_score': 0.6, 'success_rate': 0.0}
        },
        'cluster_metrics': {'silhouette_score': 0.5, 'cluster_purity': 0.75, 'neighbor_similarity_score': 0.8, 'cluster_count': 2},
        'experiment_metrics': {'16': {'dimension': 16, 'silhouette_score': 0.45, 'cluster_purity': 0.7, 'neighbor_similarity_score': 0.8}}
    }
    emb_file = settings.data_path / 'student_embeddings.json'
    emb_file.write_text(json.dumps(embeddings))

    resp = client.post('/cache/refresh')
    assert resp.status_code == 200

    profile_resp = client.get('/students/profile/1?top_k=2')
    assert profile_resp.status_code == 200
    profile_body = profile_resp.json()
    assert profile_body['student_id'] == 1
    assert profile_body['cluster_label'] == 0
    assert 'features' in profile_body
    assert isinstance(profile_body['similar_students'], list)

    similar_resp = client.get('/students/similar/1?top_k=2')
    assert similar_resp.status_code == 200
    similar_body = similar_resp.json()
    assert similar_body['student_id'] == 1
    assert isinstance(similar_body['neighbors'], list)
    assert len(similar_body['neighbors']) == 2

    clusters_resp = client.get('/students/clusters')
    assert clusters_resp.status_code == 200
    clusters_body = clusters_resp.json()
    assert clusters_body['student_count'] == 4
    assert clusters_body['cluster_metrics']['cluster_count'] == 2
    assert isinstance(clusters_body['cluster_assignments'], dict)


def test_student_embedding_experiments_endpoint(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None

    interactions = pd.DataFrame([
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.2},
        {'student_id': 2, 'topic': 'Vectors', 'score': 0.8}
    ])
    interactions_file = settings.data_path / 'student_interactions.csv'
    interactions_file.write_text(interactions.to_csv(index=False))

    embeddings = {
        'embeddings': {
            '1': [0.0, 0.0],
            '2': [0.1, 0.1]
        },
        'features': {
            '1': {'mean_score': 0.2, 'success_rate': 0.0},
            '2': {'mean_score': 0.8, 'success_rate': 1.0}
        },
        'cluster_metrics': {'silhouette_score': 0.5, 'cluster_count': 2},
        'experiment_metrics': {'16': {'silhouette_score': 0.4}, '32': {'silhouette_score': 0.5}}
    }
    emb_file = settings.data_path / 'student_embeddings.json'
    emb_file.write_text(json.dumps(embeddings))

    resp = client.get('/students/embeddings/experiments')
    assert resp.status_code == 200
    body = resp.json()
    assert body['student_count'] == 2
    assert 'experiment_metrics' in body
    assert 'cluster_metrics' in body


def test_knowledge_tracing_student_mastery_endpoints(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None
    app.state.knowledge_tracing = None

    interactions = pd.DataFrame([
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.2, 'timestamp': '2024-01-01T08:00:00'},
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.7, 'timestamp': '2024-01-02T08:00:00'},
        {'student_id': 1, 'topic': 'Attention', 'score': 0.4, 'timestamp': '2024-01-01T09:00:00'},
        {'student_id': 1, 'topic': 'Attention', 'score': 0.6, 'timestamp': '2024-01-02T09:00:00'},
        {'student_id': 2, 'topic': 'Matrices', 'score': 0.8, 'timestamp': '2024-01-01T08:30:00'}
    ])
    interactions_file = settings.data_path / 'student_interactions.csv'
    interactions_file.write_text(interactions.to_csv(index=False))

    resp = client.get('/students/1/mastery')
    assert resp.status_code == 200
    mastery_body = resp.json()
    assert mastery_body['student_id'] == 1
    assert 'Matrices' in mastery_body['mastery']
    assert 'Attention' in mastery_body['mastery']

    weak_resp = client.get('/students/1/weak_concepts?threshold=0.65&top_k=2')
    assert weak_resp.status_code == 200
    weak_body = weak_resp.json()
    assert weak_body['student_id'] == 1
    assert isinstance(weak_body['weak_concepts'], list)

    traj_resp = client.get('/students/1/trajectory?concept=Matrices')
    assert traj_resp.status_code == 200
    traj_body = traj_resp.json()
    assert traj_body['student_id'] == 1
    assert traj_body['concept'] == 'Matrices'
    assert isinstance(traj_body['trajectory'], list)

    metrics_resp = client.get('/knowledge-tracing/metrics')
    assert metrics_resp.status_code == 200
    metrics_body = metrics_resp.json()
    assert 'metrics' in metrics_body
    assert 'accuracy' in metrics_body['metrics']
    assert 'auc' in metrics_body['metrics']

    heatmap_resp = client.get('/knowledge-tracing/heatmap?max_students=2')
    assert heatmap_resp.status_code == 200
    heatmap_body = heatmap_resp.json()
    assert 'concepts' in heatmap_body
    assert 'matrix' in heatmap_body


def test_recommendations_endpoint_returns_inline_rationale(tmp_path):
    settings.data_path = tmp_path
    settings.model_checkpoint_path = tmp_path / 'checkpoints'
    settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
    settings.results_path = tmp_path / 'results'
    settings.results_path.mkdir(parents=True, exist_ok=True)
    app.state.embedding_cache = None

    interactions = pd.DataFrame([
        {'student_id': 1, 'topic': 'Matrices', 'score': 0.2},
        {'student_id': 1, 'topic': 'Vectors', 'score': 0.8},
        {'student_id': 2, 'topic': 'Matrices', 'score': 0.75},
        {'student_id': 3, 'topic': 'Vectors', 'score': 0.9},
    ])
    interactions_file = settings.data_path / 'student_interactions.csv'
    interactions_file.write_text(interactions.to_csv(index=False))

    embeddings = {
        'embeddings': {
            '1': [0.0, 0.0],
            '2': [0.1, 0.0],
            '3': [0.0, 0.1]
        },
        'features': {
            '1': {'mean_score': 0.5, 'success_rate': 0.5},
            '2': {'mean_score': 0.75, 'success_rate': 1.0},
            '3': {'mean_score': 0.9, 'success_rate': 1.0}
        }
    }
    emb_file = settings.data_path / 'student_embeddings.json'
    emb_file.write_text(json.dumps(embeddings))

    topic_embeddings = {
        'Matrices': [1.0, 0.0],
        'Vectors': [0.0, 1.0]
    }
    topic_emb_file = settings.data_path / 'embeddings.json'
    topic_emb_file.write_text(json.dumps({'topic_embeddings': topic_embeddings}))

    resp = client.get('/recommendations/1?model=hybrid&top_k=2')
    assert resp.status_code == 200
    body = resp.json()
    assert body['student_id'] == 1
    assert len(body['recommendations']) == 2
    for rec in body['recommendations']:
        assert 'reason' in rec


if __name__ == '__main__':
    test_recommend_explain_endpoint(Path('.'))
    print('OK')
