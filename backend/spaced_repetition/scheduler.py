"""Spaced Repetition Scheduler

Orchestrates SM-2, forgetting curves, and review scheduling.
Integrates with knowledge tracing for mastery-aware scheduling.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from backend.spaced_repetition.sm2 import SM2Scheduler
from backend.spaced_repetition.forgetting_curve import ForgettingCurve, RetentionPredictor
from backend.spaced_repetition.review_store import ReviewStore

logger = logging.getLogger(__name__)


class SpacedRepetitionScheduler:
    """Main spaced repetition scheduling service."""
    
    def __init__(self):
        self.sm2 = SM2Scheduler(initial_ease=2.5, min_ease=1.3)
        self.forgetting_curve = ForgettingCurve()
        self.retention_predictor = RetentionPredictor()
        self.store = ReviewStore()
    
    def initialize_for_student(
        self,
        student_id: int,
        topics: List[str],
        initial_ease: float = 2.5
    ):
        """Initialize spaced repetition tracking for a student.
        
        Args:
            student_id: Student ID
            topics: List of topics to track
            initial_ease: Starting ease factor (default 2.5)
        """
        for topic in topics:
            self.store.initialize_topic(student_id, topic, initial_ease)
    
    def record_review(
        self,
        student_id: int,
        topic: str,
        quality: int,
        mastery: float = None
    ) -> Dict:
        """
        Record a review and update SM-2 state.
        
        Args:
            student_id: Student ID
            topic: Topic reviewed
            quality: Quality of response (0-5)
            mastery: Current mastery level (optional, for integration with KT)
        
        Returns:
            Updated SM-2 state
        """
        if not (0 <= quality <= 5):
            raise ValueError("Quality must be 0-5")
        
        # Get current state
        state = self.store.get_state(student_id, topic)
        if state is None:
            self.store.initialize_topic(student_id, topic)
            state = self.store.get_state(student_id, topic)
        
        current_ease = state['ease_factor']
        repetition = state['repetition']
        interval = state['interval']
        last_review_time = datetime.now()
        
        # Update using SM-2
        next_review_time, new_ease, new_repetition = self.sm2.calculate_next_review_time(
            quality=quality,
            repetition=repetition,
            current_ease=current_ease,
            prev_interval=interval,
            last_review_time=last_review_time
        )
        
        new_interval = (next_review_time - last_review_time).days
        
        # Update store
        self.store.update_state(
            student_id=student_id,
            topic=topic,
            ease_factor=new_ease,
            repetition=new_repetition,
            interval=new_interval,
            last_review=last_review_time,
            next_review=next_review_time,
            quality=quality
        )
        
        logger.info(
            f"Recorded review: student={student_id}, topic={topic}, "
            f"quality={quality}, next_review={new_repetition}R after {new_interval}d"
        )
        
        return {
            'topic': topic,
            'quality': quality,
            'ease_factor': new_ease,
            'repetition': new_repetition,
            'interval': new_interval,
            'next_review': next_review_time.isoformat(),
        }
    
    def get_due_reviews(self, student_id: int) -> List[Dict]:
        """Get topics due for review now."""
        due_topics = self.store.get_due_reviews(student_id)
        
        reviews = []
        for topic in due_topics:
            state = self.store.get_state(student_id, topic)
            reviews.append({
                'topic': topic,
                'ease_factor': state['ease_factor'],
                'repetition': state['repetition'],
                'status': 'due',
            })
        
        return sorted(reviews, key=lambda x: -x['repetition'])  # Prioritize harder items
    
    def get_schedule(
        self,
        student_id: int,
        days_ahead: int = 7
    ) -> Dict:
        """Get review schedule for next N days."""
        scheduled = self.store.get_scheduled_reviews(student_id, days_ahead)
        
        # Group by date
        by_date = {}
        for review in scheduled:
            date_str = review['due_date'].split('T')[0]  # YYYY-MM-DD
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(review)
        
        return {
            'student_id': student_id,
            'schedule_window_days': days_ahead,
            'by_date': by_date,
            'total_reviews': sum(len(v) for v in by_date.values()),
        }
    
    def get_retention_forecast(
        self,
        student_id: int,
        topic: str,
        forecast_days: int = 30
    ) -> Dict:
        """Forecast retention over time.
        
        Args:
            student_id: Student ID
            topic: Topic to forecast
            forecast_days: Days to forecast (default 30)
        
        Returns:
            Retention curve and analysis
        """
        state = self.store.get_state(student_id, topic)
        if state is None:
            return {
                'topic': topic,
                'message': 'No review data',
                'ease_factor': 2.5,
                'strength': 0.0,
                'current_retention': 0.0,
                'retention_curve': [],
                'days_until_90_percent': None,
                'forecast_window_days': forecast_days,
            }
        
        ease_factor = state['ease_factor']
        strength = self.forgetting_curve.strength_from_ease_factor(ease_factor)
        
        # Generate curve
        curve = self.forgetting_curve.retention_curve(strength, max_days=forecast_days)
        
        # Find when it drops below 90%
        days_to_90 = self.forgetting_curve.time_to_target_retention(0.9, strength)
        
        return {
            'topic': topic,
            'ease_factor': ease_factor,
            'strength': strength,
            'current_retention': self.forgetting_curve.retention_at_time(0, strength),
            'retention_curve': [(day, retention) for day, retention in curve],
            'days_until_90_percent': days_to_90,
            'forecast_window_days': forecast_days,
        }
    
    def get_student_retention_status(
        self,
        student_id: int,
        mastery_levels: Dict[str, float] = None
    ) -> Dict:
        """Get overall retention status for all topics.
        
        Args:
            student_id: Student ID
            mastery_levels: Optional dict of topic -> mastery (from KT service)
        
        Returns:
            Retention status for all topics
        """
        topics = self.store.get_all_topics_for_student(student_id)
        mastery_levels = mastery_levels or {}
        
        status = {
            'student_id': student_id,
            'topics': [],
            'overall_retention': 0.0,
            'due_count': 0,
        }
        
        retentions = []
        
        for topic in topics:
            state = self.store.get_state(student_id, topic)
            history = self.store.get_history(student_id, topic)
            
            ease_factor = state['ease_factor']
            strength = self.forgetting_curve.strength_from_ease_factor(ease_factor)
            
            # Time since last review
            last_review = state.get('last_review')
            if last_review:
                last_review_dt = datetime.fromisoformat(last_review)
                days_since = (datetime.now() - last_review_dt).days
            else:
                days_since = 0
            
            retention = self.forgetting_curve.retention_at_time(float(days_since), strength)
            retentions.append(retention)
            
            is_due = (state['next_review'] <= datetime.now().isoformat())
            if is_due:
                status['due_count'] += 1
            
            status['topics'].append({
                'topic': topic,
                'retention': retention,
                'ease_factor': ease_factor,
                'repetition': state['repetition'],
                'mastery': mastery_levels.get(topic, 0.0),
                'due': is_due,
                'review_count': len(history),
            })
        
        if retentions:
            status['overall_retention'] = sum(retentions) / len(retentions)
        
        return status
    
    def get_statistics(self, student_id: int) -> Dict:
        """Get review statistics for a student."""
        return self.store.get_statistics(student_id)
