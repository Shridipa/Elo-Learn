"""Tests for Forgetting Curve Module"""

import pytest
import math
from datetime import datetime
from backend.spaced_repetition.forgetting_curve import ForgettingCurve, RetentionPredictor


def test_forgetting_curve_retention_calculation():
    """Test retention probability at various time points."""
    curve = ForgettingCurve()
    
    # At t=0, retention should be 1.0 (100%)
    retention = curve.retention_at_time(0, strength=1.0)
    assert retention == 1.0
    
    # Retention decreases over time
    r_1_day = curve.retention_at_time(1.0, strength=1.0)
    r_3_days = curve.retention_at_time(3.0, strength=1.0)
    r_7_days = curve.retention_at_time(7.0, strength=1.0)
    
    assert r_1_day > r_3_days > r_7_days > 0
    
    # Higher strength = slower decay
    strong_retention = curve.retention_at_time(7, strength=5.0)
    weak_retention = curve.retention_at_time(7, strength=1.0)
    assert strong_retention > weak_retention


def test_forgetting_curve_time_to_target():
    """Test inverse: time to reach target retention level."""
    curve = ForgettingCurve()
    
    # Time to reach 50% retention
    days_50 = curve.time_to_target_retention(0.5, strength=1.0)
    assert days_50 == pytest.approx(0.693, rel=0.01)  # ln(2) ≈ 0.693
    
    # Time to reach 90% retention (should be shorter)
    days_90 = curve.time_to_target_retention(0.9, strength=1.0)
    assert days_90 < days_50
    
    # With higher strength, same target takes more days
    days_50_strong = curve.time_to_target_retention(0.5, strength=2.0)
    assert days_50_strong > days_50


def test_forgetting_curve_strength_from_ease():
    """Test conversion from SM-2 ease factor to strength."""
    curve = ForgettingCurve()
    
    # Min ease factor -> low strength
    strength_min = curve.strength_from_ease_factor(1.3)
    assert strength_min > 0
    
    # Higher ease factor -> higher strength
    strength_normal = curve.strength_from_ease_factor(2.5)
    strength_high = curve.strength_from_ease_factor(3.5)
    
    assert strength_normal > strength_min
    assert strength_high > strength_normal


def test_forgetting_curve_generation():
    """Test retention curve generation for plotting."""
    curve = ForgettingCurve()
    
    curve_data = curve.retention_curve(strength=1.0, max_days=10, interval=1.0)
    
    assert len(curve_data) == 11  # 0-10 days inclusive
    
    # First point should be (0, 1.0)
    assert curve_data[0] == (0.0, 1.0)
    
    # Retention should be monotonically decreasing
    for i in range(len(curve_data) - 1):
        assert curve_data[i][1] > curve_data[i+1][1]


def test_forgetting_curve_predict_on_date():
    """Test predicting retention on a specific future date."""
    curve = ForgettingCurve()
    
    last_review = datetime(2024, 1, 1)
    target_date_1_day = datetime(2024, 1, 2)
    target_date_7_days = datetime(2024, 1, 8)
    
    retention_1_day = curve.predict_retention_on_date(
        target_date_1_day, last_review, strength=1.0
    )
    retention_7_days = curve.predict_retention_on_date(
        target_date_7_days, last_review, strength=1.0
    )
    
    assert retention_1_day > retention_7_days
    assert 0 < retention_1_day < 1
    assert 0 < retention_7_days < 1


def test_retention_predictor_status():
    """Test retention status prediction."""
    predictor = RetentionPredictor()
    
    # No review history
    status = predictor.predict_retention_status(
        'Matrices',
        [],
        mastery_level=0.7
    )
    
    assert status['topic'] == 'Matrices'
    assert status['retention'] == 0.0
    assert status['needs_review'] is True
    assert status['reason'] == 'no_reviews'


def test_retention_predictor_with_history():
    """Test retention prediction with review history."""
    predictor = RetentionPredictor()
    
    review_history = [
        {'timestamp': datetime.now(), 'quality': 4},
        {'timestamp': datetime.now(), 'quality': 4},
    ]
    
    status = predictor.predict_retention_status(
        'Calculus',
        review_history,
        mastery_level=0.8
    )
    
    assert status['topic'] == 'Calculus'
    assert status['retention'] >= 0
    assert 'days_until_threshold' in status
    assert status['review_history_length'] == 2
