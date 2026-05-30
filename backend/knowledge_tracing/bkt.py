"""Bayesian Knowledge Tracing implementation."""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class BayesianKnowledgeTracer:
    def __init__(self,
                 prior: float = 0.25,
                 learn: float = 0.15,
                 guess: float = 0.2,
                 slip: float = 0.1):
        self.prior = prior
        self.learn = learn
        self.guess = guess
        self.slip = slip

    def _update(self, p_known: float, correct: bool) -> float:
        if correct:
            p_obs = p_known * (1 - self.slip)
            denom = p_obs + (1 - p_known) * self.guess
        else:
            p_obs = p_known * self.slip
            denom = p_obs + (1 - p_known) * (1 - self.guess)

        if denom <= 0:
            posterior = 0.0
        else:
            posterior = p_obs / denom

        posterior = np.clip(posterior + (1 - posterior) * self.learn, 0.0, 1.0)
        return float(posterior)

    def estimate_trajectory(self,
                            observations: List[Dict[str, float]],
                            prior: Optional[float] = None,
                            learn: Optional[float] = None,
                            guess: Optional[float] = None,
                            slip: Optional[float] = None) -> List[float]:
        p = prior if prior is not None else self.prior
        learn = learn if learn is not None else self.learn
        guess = guess if guess is not None else self.guess
        slip = slip if slip is not None else self.slip

        trajectory = []
        for obs in observations:
            correct = bool(obs.get('is_correct', obs.get('score', 0.0) >= 0.7))
            if correct:
                p_obs = p * (1 - slip)
                denom = p_obs + (1 - p) * guess
            else:
                p_obs = p * slip
                denom = p_obs + (1 - p) * (1 - guess)
            posterior = float(p_obs / denom) if denom > 0 else 0.0
            posterior = np.clip(posterior + (1 - posterior) * learn, 0.0, 1.0)
            trajectory.append(posterior)
            p = posterior

        return trajectory

    def estimate_mastery(self,
                         observations: List[Dict[str, float]],
                         **kwargs) -> float:
        trajectory = self.estimate_trajectory(observations, **kwargs)
        return float(trajectory[-1]) if trajectory else float(kwargs.get('prior', self.prior))

    def estimate_mastery_from_df(self,
                                 student_df: pd.DataFrame,
                                 concept: str,
                                 time_column: str = 'timestamp') -> Dict[str, List[float]]:
        df = student_df[student_df['topic'] == concept].copy()
        if time_column in df.columns:
            df = df.sort_values(time_column)
        observations = df.to_dict(orient='records')
        trajectory = self.estimate_trajectory(observations)
        return {
            'concept': concept,
            'trajectory': trajectory,
            'final_mastery': trajectory[-1] if trajectory else self.prior
        }
