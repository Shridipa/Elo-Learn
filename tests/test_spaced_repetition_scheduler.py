"""Tests for Spaced Repetition Scheduler"""

import pytest
from datetime import datetime, timedelta
from backend.spaced_repetition.scheduler import SpacedRepetitionScheduler


def test_scheduler_initialization():
    """Test scheduler initialization."""
    scheduler = SpacedRepetitionScheduler()
    
    assert scheduler.sm2 is not None
    assert scheduler.forgetting_curve is not None
    assert scheduler.store is not None


def test_scheduler_initialize_for_student():
    """Test initializing topics for a student."""
    scheduler = SpacedRepetitionScheduler()
    
    topics = ['Math', 'Science', 'History']
    scheduler.initialize_for_student(1, topics)
    
    for topic in topics:
        state = scheduler.store.get_state(1, topic)
        assert state is not None
        assert state['ease_factor'] == 2.5
        assert state['repetition'] == 0


def test_scheduler_record_review_good_quality():
    """Test recording a review with good quality."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['Calculus'])
    
    result = scheduler.record_review(
        student_id=1,
        topic='Calculus',
        quality=5  # Perfect response to ensure ease factor increases
    )
    
    assert result['topic'] == 'Calculus'
    assert result['quality'] == 5
    assert result['repetition'] == 1  # First successful review -> repetition 1
    assert result['interval'] == 1  # Should be 1 day
    assert result['ease_factor'] > 2.5  # Ease factor increased (perfect response)


def test_scheduler_record_review_poor_quality():
    """Test recording a review with poor quality resets progress."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['Physics'])
    
    # Simulate multiple good reviews
    scheduler.record_review(1, 'Physics', quality=4)
    scheduler.record_review(1, 'Physics', quality=4)
    
    # Poor review should reset
    result = scheduler.record_review(1, 'Physics', quality=1)
    
    assert result['repetition'] == 1  # Reset to 1
    assert result['interval'] == 1  # Back to 1 day


def test_scheduler_get_due_reviews():
    """Test getting due reviews."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['Topic1', 'Topic2'])
    
    # Record a review to make Topic1 not due
    scheduler.record_review(1, 'Topic1', quality=4)
    
    # Topic2 should still be due (initialized with due date = now)
    due = scheduler.get_due_reviews(1)
    
    # Should have at least Topic2
    due_topics = [r['topic'] for r in due]
    assert 'Topic2' in due_topics


def test_scheduler_get_schedule():
    """Test getting review schedule."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['A', 'B', 'C'])
    
    schedule = scheduler.get_schedule(1, days_ahead=7)
    
    assert schedule['student_id'] == 1
    assert schedule['schedule_window_days'] == 7
    assert 'by_date' in schedule


def test_scheduler_retention_forecast():
    """Test retention forecast generation."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['Biology'])
    
    # Record a few reviews to build history
    scheduler.record_review(1, 'Biology', quality=4)
    scheduler.record_review(1, 'Biology', quality=4)
    
    forecast = scheduler.get_retention_forecast(1, 'Biology', forecast_days=30)
    
    assert forecast['topic'] == 'Biology'
    assert forecast['ease_factor'] > 0
    assert forecast['current_retention'] >= 0
    assert 'retention_curve' in forecast
    assert len(forecast['retention_curve']) > 0


def test_scheduler_retention_status():
    """Test overall retention status."""
    scheduler = SpacedRepetitionScheduler()
    
    topics = ['A', 'B', 'C']
    scheduler.initialize_for_student(1, topics)
    
    # Record reviews
    for topic in topics:
        scheduler.record_review(1, topic, quality=4)
    
    status = scheduler.get_student_retention_status(1)
    
    assert status['student_id'] == 1
    assert 'overall_retention' in status
    assert 'due_count' in status
    assert len(status['topics']) == 3


def test_scheduler_statistics():
    """Test review statistics."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['X', 'Y'])
    
    # Record reviews
    scheduler.record_review(1, 'X', quality=5)
    scheduler.record_review(1, 'Y', quality=2)
    
    stats = scheduler.get_statistics(1)
    
    assert stats['total_topics'] == 2
    assert stats['avg_ease_factor'] > 1.3
    assert stats['avg_repetitions'] >= 1


def test_scheduler_quality_validation():
    """Test that invalid quality raises error."""
    scheduler = SpacedRepetitionScheduler()
    scheduler.initialize_for_student(1, ['Math'])
    
    with pytest.raises(ValueError):
        scheduler.record_review(1, 'Math', quality=6)
    
    with pytest.raises(ValueError):
        scheduler.record_review(1, 'Math', quality=-1)
