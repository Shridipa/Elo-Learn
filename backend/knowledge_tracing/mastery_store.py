"""In-memory mastery store for knowledge tracing."""
from typing import Dict, Any, List, Optional
import pandas as pd


class MasteryStore:
    def __init__(self):
        self.mastery_by_student: Dict[int, Dict[str, float]] = {}
        self.trajectory_by_student: Dict[int, Dict[str, List[float]]] = {}
        self.concepts: List[str] = []

    def build(self, student_mastery: Dict[int, Dict[str, float]],
              trajectories: Dict[int, Dict[str, List[float]]],
              concepts: Optional[List[str]] = None) -> None:
        self.mastery_by_student = student_mastery
        self.trajectory_by_student = trajectories
        self.concepts = concepts or sorted({c for v in student_mastery.values() for c in v.keys()})

    def get_mastery(self, student_id: int) -> Dict[str, float]:
        return self.mastery_by_student.get(student_id, {})

    def get_weak_concepts(self, student_id: int, threshold: float = 0.65, top_k: int = 5) -> List[str]:
        mastery = self.get_mastery(student_id)
        if not mastery:
            return []
        weak = sorted(mastery.items(), key=lambda x: x[1])
        return [concept for concept, _ in weak[:top_k] if _ < threshold]

    def get_trajectory(self, student_id: int, concept: str) -> List[float]:
        return self.trajectory_by_student.get(student_id, {}).get(concept, [])

    def get_heatmap(self, max_students: int = 20) -> Dict[str, Any]:
        student_ids = sorted(self.mastery_by_student.keys())[:max_students]
        matrix = [
            {'student_id': student_id, **{concept: self.mastery_by_student[student_id].get(concept, 0.0) for concept in self.concepts}}
            for student_id in student_ids
        ]
        return {
            'concepts': self.concepts,
            'students': student_ids,
            'matrix': matrix
        }
