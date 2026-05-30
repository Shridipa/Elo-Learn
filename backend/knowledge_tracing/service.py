"""Service layer for knowledge tracing operations."""
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from .bkt import BayesianKnowledgeTracer
from .evaluator import compute_metrics
from .mastery_store import MasteryStore


class KnowledgeTracingService:
    def __init__(self,
                 interactions_path: Path,
                 concept_graph_path: Optional[Path] = None):
        self.interactions_path = interactions_path
        self.concept_graph_path = concept_graph_path
        self.tracer = BayesianKnowledgeTracer()
        self.store = MasteryStore()
        self.metrics: Dict[str, Any] = {}
        self._refresh()

    def _load_interactions(self) -> pd.DataFrame:
        df = pd.read_csv(self.interactions_path)
        if 'timestamp' in df.columns:
            df = df.sort_values(['student_id', 'timestamp'])
        return df

    def _refresh(self) -> None:
        df = self._load_interactions()
        student_mastery: Dict[int, Dict[str, float]] = {}
        trajectories: Dict[int, Dict[str, List[float]]] = {}
        concepts = sorted(df['topic'].dropna().unique().tolist())

        for student_id in df['student_id'].unique():
            student_df = df[df['student_id'] == student_id]
            mastery = {}
            trajectory_map: Dict[str, List[float]] = {}
            for concept in concepts:
                concept_df = student_df[student_df['topic'] == concept]
                if concept_df.empty:
                    continue
                result = self.tracer.estimate_mastery_from_df(concept_df, concept)
                mastery[concept] = float(result['final_mastery'])
                trajectory_map[concept] = result['trajectory']
            student_mastery[int(student_id)] = mastery
            trajectories[int(student_id)] = trajectory_map

        self.store.build(student_mastery, trajectories, concepts=concepts)
        self.metrics = self._compute_metrics(df)

    def refresh(self) -> None:
        self._refresh()

    def get_student_mastery(self, student_id: int) -> Dict[str, float]:
        return self.store.get_mastery(student_id)

    def get_weak_concepts(self, student_id: int, threshold: float = 0.65, top_k: int = 5) -> List[str]:
        return self.store.get_weak_concepts(student_id, threshold=threshold, top_k=top_k)

    def get_trajectory(self, student_id: int, concept: str) -> List[float]:
        return self.store.get_trajectory(student_id, concept)

    def get_heatmap(self, max_students: int = 20) -> Dict[str, Any]:
        return self.store.get_heatmap(max_students=max_students)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        predictions: List[float] = []
        labels: List[int] = []
        for student_id in df['student_id'].unique():
            student_df = df[df['student_id'] == student_id]
            for concept in student_df['topic'].unique():
                concept_df = student_df[student_df['topic'] == concept]
                trajectory = self.tracer.estimate_trajectory(concept_df.to_dict(orient='records'))
                if not trajectory:
                    continue
                predictions.extend(trajectory)
                labels.extend([int(bool(score >= 0.7)) for score in concept_df['score'].tolist()])
        return compute_metrics(predictions, labels)
