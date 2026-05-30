"""Forgetting Curve and Retention Modeling

Implements Ebbinghaus's forgetting curve and retention prediction models
based on spacing and learner strength.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ForgettingCurve:
    """Ebbinghaus Forgetting Curve
    
    Models memory retention over time using exponential decay.
    Incorporates spaced repetition impact via strength factor.
    
    Retention formula:
    R(t) = e^(-t / S)
    
    Where:
    - R(t) = retention at time t (0-1)
    - t = time since last review (days)
    - S = strength factor (higher = slower decay)
    """
    
    def __init__(self, base_strength: float = 1.0):
        """
        Args:
            base_strength: Base strength for new items (default 1.0)
        """
        self.base_strength = base_strength
    
    def strength_from_ease_factor(self, ease_factor: float) -> float:
        """
        Convert SM-2 ease factor to retention strength.
        
        Higher ease factor -> slower forgetting
        
        Args:
            ease_factor: SM-2 ease factor (typically 1.3-2.5+)
        
        Returns:
            Strength S in forgetting curve (1.0-5.0+)
        """
        # Map ease_factor [1.3, 2.5] -> strength [1.0, 3.0]
        # Ease 1.3 = very hard to remember (fast decay)
        # Ease 2.5 = normal difficulty (moderate decay)
        # Ease 3.5+ = very easy (slow decay)
        
        return max(1.0, (ease_factor - 0.3) * 1.2)
    
    def retention_at_time(
        self,
        days_since_review: float,
        strength: float
    ) -> float:
        """
        Calculate retention probability at a given time.
        
        R(t) = e^(-t / S)
        
        Args:
            days_since_review: Days elapsed since last review
            strength: Strength factor (higher = slower decay)
        
        Returns:
            Retention probability (0.0 to 1.0)
        """
        if days_since_review < 0:
            return 1.0
        
        if strength <= 0:
            strength = self.base_strength
        
        retention = math.exp(-days_since_review / strength)
        return max(0.0, min(1.0, retention))
    
    def time_to_target_retention(
        self,
        target_retention: float,
        strength: float
    ) -> float:
        """
        Calculate days until retention drops to target level.
        
        Inverse of retention_at_time:
        t = -S * ln(R)
        
        Args:
            target_retention: Target retention level (0.0-1.0)
            strength: Strength factor
        
        Returns:
            Days until reaching target retention
        """
        if target_retention >= 1.0:
            return 0.0
        
        if target_retention <= 0.0:
            return float('inf')
        
        if strength <= 0:
            strength = self.base_strength
        
        days = -strength * math.log(target_retention)
        return max(0.0, days)
    
    def optimal_review_point(
        self,
        strength: float,
        target_retention: float = 0.9
    ) -> float:
        """
        Calculate optimal time to review (when retention reaches threshold).
        
        Args:
            strength: Strength factor
            target_retention: When to review (default 90%)
        
        Returns:
            Days until optimal review point
        """
        return self.time_to_target_retention(target_retention, strength)
    
    def retention_curve(
        self,
        strength: float,
        max_days: int = 365,
        interval: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Generate retention curve data for plotting.
        
        Args:
            strength: Strength factor
            max_days: Maximum days to calculate (default 365)
            interval: Step size in days (default 1.0)
        
        Returns:
            List of (days, retention) tuples
        """
        curve = []
        days = 0.0
        
        while days <= max_days:
            retention = self.retention_at_time(days, strength)
            curve.append((days, retention))
            days += interval
        
        return curve
    
    def predict_retention_on_date(
        self,
        target_date: datetime,
        last_review_date: datetime,
        strength: float
    ) -> float:
        """
        Predict retention on a specific future date.
        
        Args:
            target_date: Date to predict retention for
            last_review_date: When the last review occurred
            strength: Strength factor
        
        Returns:
            Predicted retention probability (0.0-1.0)
        """
        delta = (target_date - last_review_date).days
        return self.retention_at_time(float(delta), strength)


class RetentionPredictor:
    """Predict overall retention based on multiple reviews."""
    
    def __init__(self):
        self.forgetting_curve = ForgettingCurve()
    
    def predict_retention_status(
        self,
        topic: str,
        review_history: List[Dict],
        mastery_level: float = 0.7,
        mastery_decay_per_day: float = 0.01
    ) -> Dict:
        """
        Predict retention status based on review history and mastery.
        
        Args:
            topic: The concept being reviewed
            review_history: List of review records with timestamp and quality
            mastery_level: Current mastery (0-1)
            mastery_decay_per_day: Daily decay rate (default 0.01 = 1% per day)
        
        Returns:
            Dict with retention metrics and recommendation
        """
        if not review_history:
            return {
                'topic': topic,
                'retention': 0.0,
                'needs_review': True,
                'days_until_review': 0,
                'reason': 'no_reviews'
            }
        
        last_review = review_history[-1]['timestamp']
        days_since = (datetime.now() - last_review).days
        
        # Decay mastery over time
        decayed_mastery = max(0.0, mastery_level - (days_since * mastery_decay_per_day))
        
        # Estimate strength from review history
        avg_quality = sum(r.get('quality', 3) for r in review_history) / len(review_history)
        ease_factor = 1.3 + (avg_quality - 0) * 0.3  # Map quality 0-5 to ease 1.3-2.8
        strength = self.forgetting_curve.strength_from_ease_factor(ease_factor)
        
        retention = self.forgetting_curve.retention_at_time(float(days_since), strength)
        
        return {
            'topic': topic,
            'retention': retention,
            'mastery': decayed_mastery,
            'needs_review': retention < 0.9 or decayed_mastery < 0.6,
            'days_until_threshold': self.forgetting_curve.time_to_target_retention(0.9, strength),
            'review_history_length': len(review_history),
            'reason': 'low_retention' if retention < 0.9 else 'mastery_decay'
        }
