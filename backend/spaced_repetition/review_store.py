"""Review Store for Spaced Repetition

Manages persistent storage and retrieval of review data per student.
Tracks ease factors, repetition counts, and review history.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ReviewRecord:
    """Single review record for a concept."""
    
    def __init__(
        self,
        student_id: int,
        topic: str,
        quality: int,
        timestamp: datetime = None,
        ease_factor: float = 2.5,
        repetition: int = 1,
        interval: int = 0
    ):
        self.student_id = student_id
        self.topic = topic
        self.quality = quality
        self.timestamp = timestamp or datetime.now()
        self.ease_factor = ease_factor
        self.repetition = repetition
        self.interval = interval
    
    def to_dict(self) -> Dict:
        return {
            'student_id': self.student_id,
            'topic': self.topic,
            'quality': self.quality,
            'timestamp': self.timestamp.isoformat(),
            'ease_factor': self.ease_factor,
            'repetition': self.repetition,
            'interval': self.interval,
        }


class ReviewStore:
    """In-memory review store for managing spaced repetition data.
    
    In production, this would be backed by a database.
    Current implementation uses in-memory dict.
    
    Structure:
    {
        student_id: {
            topic: {
                'ease_factor': float,
                'repetition': int,
                'interval': int,
                'last_review': datetime,
                'next_review': datetime,
                'history': [ReviewRecord, ...]
            }
        }
    }
    """
    
    def __init__(self):
        self.store: Dict[int, Dict[str, Dict]] = {}
    
    def _ensure_student(self, student_id: int):
        """Ensure student entry exists."""
        if student_id not in self.store:
            self.store[student_id] = {}
    
    def _ensure_topic(self, student_id: int, topic: str):
        """Ensure topic entry exists for student."""
        self._ensure_student(student_id)
        if topic not in self.store[student_id]:
            self.store[student_id][topic] = {
                'ease_factor': 2.5,
                'repetition': 0,
                'interval': 0,
                'last_review': None,
                'next_review': datetime.now(),
                'history': [],
            }
    
    def get_state(self, student_id: int, topic: str) -> Optional[Dict]:
        """Get current SM-2 state for a topic."""
        self._ensure_topic(student_id, topic)
        state = self.store[student_id][topic].copy()
        # Convert datetime objects to ISO strings for serialization
        if state['last_review']:
            state['last_review'] = state['last_review'].isoformat()
        if state['next_review']:
            state['next_review'] = state['next_review'].isoformat()
        state.pop('history', None)  # Don't include full history in state
        return state
    
    def update_state(
        self,
        student_id: int,
        topic: str,
        ease_factor: float,
        repetition: int,
        interval: int,
        last_review: datetime,
        next_review: datetime,
        quality: int = None
    ):
        """Update SM-2 state after a review."""
        self._ensure_topic(student_id, topic)
        state = self.store[student_id][topic]
        
        state['ease_factor'] = ease_factor
        state['repetition'] = repetition
        state['interval'] = interval
        state['last_review'] = last_review
        state['next_review'] = next_review
        
        # Add to history
        if quality is not None:
            record = ReviewRecord(
                student_id=student_id,
                topic=topic,
                quality=quality,
                timestamp=last_review,
                ease_factor=ease_factor,
                repetition=repetition,
                interval=interval,
            )
            state['history'].append(record)
    
    def get_history(self, student_id: int, topic: str) -> List[Dict]:
        """Get review history for a topic."""
        self._ensure_topic(student_id, topic)
        records = self.store[student_id][topic]['history']
        return [r.to_dict() for r in records]
    
    def get_due_reviews(self, student_id: int) -> List[str]:
        """Get list of topics due for review (next_review <= now)."""
        self._ensure_student(student_id)
        now = datetime.now()
        due = []
        
        for topic, state in self.store[student_id].items():
            next_review = state.get('next_review')
            if next_review and next_review <= now:
                due.append(topic)
        
        return due
    
    def get_scheduled_reviews(
        self,
        student_id: int,
        days_ahead: int = 7
    ) -> List[Dict]:
        """Get reviews scheduled for the next N days."""
        self._ensure_student(student_id)
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)
        
        reviews = []
        for topic, state in self.store[student_id].items():
            next_review = state.get('next_review')
            if next_review and now <= next_review <= cutoff:
                reviews.append({
                    'topic': topic,
                    'due_date': next_review.isoformat(),
                    'ease_factor': state['ease_factor'],
                    'repetition': state['repetition'],
                    'days_until_due': (next_review - now).days,
                })
        
        # Sort by due date
        reviews.sort(key=lambda x: x['due_date'])
        return reviews
    
    def get_all_topics_for_student(self, student_id: int) -> List[str]:
        """Get all topics tracked for a student."""
        self._ensure_student(student_id)
        return list(self.store[student_id].keys())
    
    def initialize_topic(
        self,
        student_id: int,
        topic: str,
        initial_ease: float = 2.5
    ):
        """Initialize tracking for a new topic."""
        self._ensure_topic(student_id, topic)
        state = self.store[student_id][topic]
        state['ease_factor'] = initial_ease
        state['repetition'] = 0
        state['interval'] = 0
        state['last_review'] = None
        state['next_review'] = datetime.now()  # First review is due immediately
    
    def get_statistics(self, student_id: int) -> Dict:
        """Get review statistics for a student."""
        self._ensure_student(student_id)
        topics = self.store[student_id]
        
        if not topics:
            return {
                'total_topics': 0,
                'due_reviews': 0,
                'avg_ease_factor': 0.0,
                'avg_repetitions': 0.0,
            }
        
        due_count = len(self.get_due_reviews(student_id))
        ease_factors = [t['ease_factor'] for t in topics.values()]
        repetitions = [t['repetition'] for t in topics.values()]
        
        return {
            'total_topics': len(topics),
            'due_reviews': due_count,
            'avg_ease_factor': sum(ease_factors) / len(ease_factors),
            'avg_repetitions': sum(repetitions) / len(repetitions),
            'topics_list': list(topics.keys()),
        }
