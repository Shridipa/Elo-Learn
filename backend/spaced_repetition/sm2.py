"""SM-2 Algorithm Implementation

Implements the SuperMemo 2 spaced repetition scheduling algorithm.
Optimizes review intervals based on item difficulty and learner performance.
"""

from typing import Dict, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SM2Scheduler:
    """SM-2 Spaced Repetition Scheduler
    
    Manages ease factors and calculates optimal review intervals.
    
    Args:
        initial_ease: Starting ease factor (typically 2.5)
        min_ease: Minimum allowed ease factor (typically 1.3)
    """
    
    def __init__(self, initial_ease: float = 2.5, min_ease: float = 1.3):
        self.initial_ease = initial_ease
        self.min_ease = min_ease
    
    def calculate_ease_factor(
        self,
        current_ease: float,
        quality: int
    ) -> float:
        """
        Calculate new ease factor based on response quality.
        
        SM-2 Formula:
        EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        
        Args:
            current_ease: Current ease factor (typically 1.3 to 2.5+)
            quality: Quality of response (0-5)
                0: Complete blackout, total recall failure
                1: Incorrect response; correct one remembered
                2: Incorrect response; correct one easily remembered
                3: Correct response with serious difficulty
                4: Correct response after some hesitation
                5: Perfect response
        
        Returns:
            New ease factor (min 1.3)
        """
        if not (0 <= quality <= 5):
            raise ValueError("Quality must be between 0 and 5")
        
        new_ease = current_ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        return max(self.min_ease, new_ease)
    
    def calculate_interval(
        self,
        repetition: int,
        ease_factor: float,
        prev_interval: int = 0
    ) -> int:
        """
        Calculate review interval (days) based on SM-2 algorithm.
        
        Intervals follow pattern:
        - 1st review: 1 day
        - 2nd review: 3 days
        - 3rd+ review: previous_interval * ease_factor
        
        Args:
            repetition: Number of successful reviews (1-indexed)
            ease_factor: Current ease factor
            prev_interval: Previous interval in days (for repetition >= 3)
        
        Returns:
            Days until next review
        """
        if repetition <= 0:
            raise ValueError("Repetition must be >= 1")
        
        if repetition == 1:
            return 1
        elif repetition == 2:
            return 3
        else:
            # For 3+, multiply previous interval by ease factor
            new_interval = int(prev_interval * ease_factor)
            return max(1, new_interval)
    
    def calculate_next_review_time(
        self,
        quality: int,
        repetition: int,
        current_ease: float,
        prev_interval: int = 0,
        last_review_time: datetime = None
    ) -> Tuple[datetime, float, int]:
        """
        Calculate next review time and update tracking info.
        
        Args:
            quality: Quality of response (0-5)
            repetition: Current repetition count
            current_ease: Current ease factor
            prev_interval: Previous interval (days)
            last_review_time: When the last review occurred (default: now)
        
        Returns:
            Tuple of:
            - next_review_time (datetime)
            - new_ease_factor (float)
            - new_repetition_count (int)
        """
        if last_review_time is None:
            last_review_time = datetime.now()
        
        # Update ease factor
        new_ease = self.calculate_ease_factor(current_ease, quality)
        
        # If quality is poor (< 3), reset repetition count
        if quality < 3:
            new_repetition = 1
            interval = 1  # Reset to 1 day
        else:
            new_repetition = repetition + 1
            interval = self.calculate_interval(new_repetition, new_ease, prev_interval)
        
        # Calculate next review datetime
        next_review = last_review_time + timedelta(days=interval)
        
        return next_review, new_ease, new_repetition
    
    def get_initial_state(self) -> Dict:
        """Get initial SM-2 state for a new item."""
        return {
            'ease_factor': self.initial_ease,
            'repetition': 0,
            'interval': 0,
            'last_review': None,
            'next_review': datetime.now(),
        }
