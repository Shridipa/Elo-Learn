"""Tests for Review Store Module"""

import pytest
from datetime import datetime, timedelta
from backend.spaced_repetition.review_store import ReviewStore


def test_review_store_initialization():
    """Test review store initialization."""
    store = ReviewStore()
    
    assert store.store == {}


def test_review_store_ensure_student():
    """Test student creation."""
    store = ReviewStore()
    store._ensure_student(1)
    
    assert 1 in store.store
    assert store.store[1] == {}


def test_review_store_ensure_topic():
    """Test topic creation for a student."""
    store = ReviewStore()
    store._ensure_topic(1, 'Calculus')
    
    assert 1 in store.store
    assert 'Calculus' in store.store[1]
    
    state = store.store[1]['Calculus']
    assert state['ease_factor'] == 2.5
    assert state['repetition'] == 0
    assert state['interval'] == 0
    assert state['last_review'] is None
    assert isinstance(state['history'], list)


def test_review_store_get_state():
    """Test getting SM-2 state."""
    store = ReviewStore()
    store._ensure_topic(1, 'Matrices')
    
    state = store.get_state(1, 'Matrices')
    
    assert state is not None
    assert 'ease_factor' in state
    assert 'repetition' in state
    assert 'interval' in state
    assert state['ease_factor'] == 2.5
    assert state['repetition'] == 0


def test_review_store_update_state():
    """Test updating SM-2 state."""
    store = ReviewStore()
    store._ensure_topic(1, 'Python')
    
    now = datetime.now()
    next_review = now + timedelta(days=3)
    
    store.update_state(
        student_id=1,
        topic='Python',
        ease_factor=2.6,
        repetition=2,
        interval=3,
        last_review=now,
        next_review=next_review,
        quality=4
    )
    
    state = store.get_state(1, 'Python')
    assert state['ease_factor'] == 2.6
    assert state['repetition'] == 2
    assert state['interval'] == 3


def test_review_store_get_history():
    """Test retrieving review history."""
    store = ReviewStore()
    store._ensure_topic(1, 'Algebra')
    
    now = datetime.now()
    next_review = now + timedelta(days=1)
    
    store.update_state(
        student_id=1,
        topic='Algebra',
        ease_factor=2.5,
        repetition=1,
        interval=1,
        last_review=now,
        next_review=next_review,
        quality=5
    )
    
    history = store.get_history(1, 'Algebra')
    
    assert len(history) == 1
    assert history[0]['topic'] == 'Algebra'
    assert history[0]['quality'] == 5


def test_review_store_get_due_reviews():
    """Test getting due reviews."""
    store = ReviewStore()
    
    # Initialize topic with past due date
    store._ensure_topic(1, 'Topic1')
    store.store[1]['Topic1']['next_review'] = datetime.now() - timedelta(days=1)
    
    # Initialize topic with future due date
    store._ensure_topic(1, 'Topic2')
    store.store[1]['Topic2']['next_review'] = datetime.now() + timedelta(days=1)
    
    due = store.get_due_reviews(1)
    
    assert 'Topic1' in due
    assert 'Topic2' not in due


def test_review_store_get_scheduled_reviews():
    """Test getting scheduled reviews within a timeframe."""
    store = ReviewStore()
    
    now = datetime.now()
    
    # Topic due in 2 days
    store._ensure_topic(1, 'TopicA')
    store.store[1]['TopicA']['next_review'] = now + timedelta(days=2)
    
    # Topic due in 10 days (outside 7-day window)
    store._ensure_topic(1, 'TopicB')
    store.store[1]['TopicB']['next_review'] = now + timedelta(days=10)
    
    scheduled = store.get_scheduled_reviews(1, days_ahead=7)
    
    assert len(scheduled) == 1
    assert scheduled[0]['topic'] == 'TopicA'
    # days_until_due might be 1 or 2 depending on exact timing
    assert scheduled[0]['days_until_due'] >= 1


def test_review_store_statistics():
    """Test getting review statistics."""
    store = ReviewStore()
    
    store._ensure_topic(1, 'Math')
    store.store[1]['Math']['ease_factor'] = 2.5
    store.store[1]['Math']['repetition'] = 3
    
    store._ensure_topic(1, 'Science')
    store.store[1]['Science']['ease_factor'] = 2.3
    store.store[1]['Science']['repetition'] = 2
    
    stats = store.get_statistics(1)
    
    assert stats['total_topics'] == 2
    assert stats['avg_ease_factor'] == pytest.approx(2.4, rel=0.1)
    assert stats['avg_repetitions'] == 2.5


def test_review_store_multiple_students():
    """Test handling multiple students."""
    store = ReviewStore()
    
    store._ensure_topic(1, 'Topic1')
    store._ensure_topic(2, 'Topic2')
    
    assert 1 in store.store
    assert 2 in store.store
    assert 'Topic1' in store.store[1]
    assert 'Topic2' in store.store[2]
