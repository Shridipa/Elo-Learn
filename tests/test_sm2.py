"""Tests for SM-2 Algorithm"""

import pytest
from datetime import datetime, timedelta
from backend.spaced_repetition.sm2 import SM2Scheduler


def test_sm2_ease_factor_calculation():
    """Test ease factor updates based on quality."""
    scheduler = SM2Scheduler()
    
    # Test with quality 5 (perfect): ease increases
    new_ease = scheduler.calculate_ease_factor(2.5, quality=5)
    assert new_ease > 2.5
    
    # Test with quality 0 (complete failure): ease decreases
    new_ease = scheduler.calculate_ease_factor(2.5, quality=0)
    assert new_ease < 2.5
    
    # Test with quality 3 (difficult but correct): ease stays roughly same
    new_ease = scheduler.calculate_ease_factor(2.5, quality=3)
    assert 2.3 < new_ease < 2.7  # More generous tolerance
    
    # Test minimum ease factor constraint
    new_ease = scheduler.calculate_ease_factor(1.3, quality=0)
    assert new_ease >= 1.3


def test_sm2_intervals():
    """Test interval calculations follow SM-2 pattern."""
    scheduler = SM2Scheduler()
    
    # 1st review: 1 day
    interval_1 = scheduler.calculate_interval(1, ease_factor=2.5)
    assert interval_1 == 1
    
    # 2nd review: 3 days
    interval_2 = scheduler.calculate_interval(2, ease_factor=2.5)
    assert interval_2 == 3
    
    # 3rd review: depends on ease factor
    interval_3 = scheduler.calculate_interval(3, ease_factor=2.5, prev_interval=3)
    assert interval_3 == 7  # 3 * 2.5 ≈ 7.5 -> 7
    
    # With higher ease factor, longer intervals
    interval_3_high = scheduler.calculate_interval(3, ease_factor=3.0, prev_interval=3)
    assert interval_3_high > interval_3


def test_sm2_next_review_time():
    """Test next review time calculation."""
    scheduler = SM2Scheduler()
    last_review = datetime(2024, 1, 1)
    
    # Good quality (4): advance to next repetition
    next_time, new_ease, new_rep = scheduler.calculate_next_review_time(
        quality=4,
        repetition=1,
        current_ease=2.5,
        prev_interval=0,
        last_review_time=last_review
    )
    assert new_rep == 2
    assert (next_time - last_review).days == 3
    
    # Poor quality (1): reset to repetition 1
    next_time, new_ease, new_rep = scheduler.calculate_next_review_time(
        quality=1,
        repetition=3,
        current_ease=2.5,
        prev_interval=7,
        last_review_time=last_review
    )
    assert new_rep == 1
    assert (next_time - last_review).days == 1


def test_sm2_quality_validation():
    """Test that invalid quality raises error."""
    scheduler = SM2Scheduler()
    
    with pytest.raises(ValueError):
        scheduler.calculate_ease_factor(2.5, quality=-1)
    
    with pytest.raises(ValueError):
        scheduler.calculate_ease_factor(2.5, quality=6)


def test_sm2_interval_validation():
    """Test that invalid repetition raises error."""
    scheduler = SM2Scheduler()
    
    with pytest.raises(ValueError):
        scheduler.calculate_interval(0, ease_factor=2.5)
    
    with pytest.raises(ValueError):
        scheduler.calculate_interval(-1, ease_factor=2.5)


def test_sm2_initial_state():
    """Test initial SM-2 state for new items."""
    scheduler = SM2Scheduler(initial_ease=2.5)
    
    initial = scheduler.get_initial_state()
    
    assert initial['ease_factor'] == 2.5
    assert initial['repetition'] == 0
    assert initial['interval'] == 0
    assert initial['last_review'] is None
    assert 'next_review' in initial
