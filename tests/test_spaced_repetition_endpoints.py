"""Tests for Spaced Repetition Endpoints"""

import sys
import pathlib
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backend.main import app
from backend.spaced_repetition.scheduler import SpacedRepetitionScheduler


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def setup_scheduler():
    """Setup scheduler with test data."""
    scheduler = SpacedRepetitionScheduler()
    app.state.sr_scheduler = scheduler
    
    # Initialize topics for students 1 and 2
    topics = ['Matrices', 'Calculus', 'Linear Algebra']
    scheduler.initialize_for_student(1, topics)
    scheduler.initialize_for_student(2, topics)
    
    # Record some reviews for student 1
    scheduler.record_review(1, 'Matrices', quality=4)
    scheduler.record_review(1, 'Calculus', quality=3)
    
    return scheduler


def test_get_due_reviews_endpoint(client, setup_scheduler):
    """Test GET /reviews/due/{student_id}"""
    response = client.get('/reviews/due/1')
    
    assert response.status_code == 200
    data = response.json()
    assert 'student_id' in data
    assert data['student_id'] == 1
    assert 'due_reviews' in data
    assert isinstance(data['due_reviews'], list)
    assert 'total_due' in data


def test_get_due_reviews_nonexistent_student(client, setup_scheduler):
    """Test GET /reviews/due/{student_id} for non-existent student."""
    response = client.get('/reviews/due/999')
    
    # Should return empty list, not error
    assert response.status_code == 200
    data = response.json()
    assert data['total_due'] == 0


def test_complete_review_endpoint(client, setup_scheduler):
    """Test POST /reviews/complete"""
    request_body = {
        'student_id': 1,
        'topic': 'Matrices',
        'quality': 4,
        'mastery': 0.7
    }
    
    response = client.post('/reviews/complete', json=request_body)
    
    assert response.status_code == 200
    data = response.json()
    assert data['topic'] == 'Matrices'
    assert data['quality'] == 4
    assert 'ease_factor' in data
    assert 'repetition' in data
    assert 'next_review' in data


def test_complete_review_invalid_quality(client, setup_scheduler):
    """Test POST /reviews/complete with invalid quality."""
    request_body = {
        'student_id': 1,
        'topic': 'Matrices',
        'quality': 10  # Invalid
    }
    
    response = client.post('/reviews/complete', json=request_body)
    
    # Pydantic validation should reject this
    assert response.status_code == 422


def test_get_review_schedule_endpoint(client, setup_scheduler):
    """Test GET /reviews/schedule/{student_id}"""
    response = client.get('/reviews/schedule/1?days_ahead=7')
    
    assert response.status_code == 200
    data = response.json()
    assert 'student_id' in data
    assert data['student_id'] == 1
    assert 'schedule_window_days' in data
    assert data['schedule_window_days'] == 7
    assert 'by_date' in data
    assert isinstance(data['by_date'], dict)


def test_get_review_schedule_custom_window(client, setup_scheduler):
    """Test GET /reviews/schedule/{student_id} with custom window."""
    response = client.get('/reviews/schedule/1?days_ahead=30')
    
    assert response.status_code == 200
    data = response.json()
    assert data['schedule_window_days'] == 30


def test_get_retention_forecast_endpoint(client, setup_scheduler):
    """Test GET /reviews/retention/{student_id}/{topic}"""
    response = client.get('/reviews/retention/1/Matrices?forecast_days=30')
    
    assert response.status_code == 200
    data = response.json()
    assert data['student_id'] == 1
    assert data['topic'] == 'Matrices'
    assert 'ease_factor' in data
    assert 'retention_curve' in data
    assert 'days_until_90_percent' in data
    assert len(data['retention_curve']) > 0


def test_get_retention_forecast_no_data(client):
    """Test retention forecast for topic with no explicit reviews (but accessed once)."""
    # Create a new scheduler without setup - this doesn't use the fixture
    from backend.spaced_repetition.scheduler import SpacedRepetitionScheduler
    scheduler = SpacedRepetitionScheduler()
    app.state.sr_scheduler = scheduler
    
    # Access a topic for the first time (will auto-create with default state)
    response = client.get('/reviews/retention/99/NewTopic?forecast_days=30')
    
    assert response.status_code == 200
    data = response.json()
    assert data['topic'] == 'NewTopic'
    # When a topic is first accessed, it has default values and generates a retention curve
    # The ease_factor should be 2.5 (default) and retention curve should exist
    assert data['ease_factor'] == 2.5
    assert len(data['retention_curve']) > 0  # Default curve generated


def test_get_retention_status_endpoint(client, setup_scheduler):
    """Test GET /reviews/status/{student_id}"""
    response = client.get('/reviews/status/1')
    
    assert response.status_code == 200
    data = response.json()
    assert data['student_id'] == 1
    assert 'overall_retention' in data
    assert 'due_count' in data
    assert 'topics' in data
    assert isinstance(data['topics'], list)


def test_get_review_statistics_endpoint(client, setup_scheduler):
    """Test GET /reviews/statistics/{student_id}"""
    response = client.get('/reviews/statistics/1')
    
    assert response.status_code == 200
    data = response.json()
    assert data['student_id'] == 1
    assert 'statistics' in data
    assert 'total_topics' in data['statistics']
    assert 'avg_ease_factor' in data['statistics']
    assert 'avg_repetitions' in data['statistics']


def test_endpoints_without_scheduler(client):
    """Test endpoints when scheduler is not initialized."""
    # Remove scheduler from app state
    app.state.sr_scheduler = None
    
    response = client.get('/reviews/due/1')
    
    assert response.status_code == 503
    assert 'unavailable' in response.json()['detail'].lower()


def test_review_workflow(client, setup_scheduler):
    """Test a complete review workflow."""
    student_id = 2
    topic = 'Matrices'
    
    # 1. Check due reviews
    response = client.get(f'/reviews/due/{student_id}')
    assert response.status_code == 200
    due_before = response.json()['total_due']
    
    # 2. Complete a review
    response = client.post('/reviews/complete', json={
        'student_id': student_id,
        'topic': topic,
        'quality': 5
    })
    assert response.status_code == 200
    result = response.json()
    assert result['quality'] == 5
    
    # 3. Check schedule
    response = client.get(f'/reviews/schedule/{student_id}?days_ahead=7')
    assert response.status_code == 200
    schedule = response.json()
    assert 'by_date' in schedule
    
    # 4. Get retention forecast
    response = client.get(f'/reviews/retention/{student_id}/{topic}')
    assert response.status_code == 200
    forecast = response.json()
    assert forecast['topic'] == topic
