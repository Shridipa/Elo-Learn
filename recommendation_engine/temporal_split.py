"""Temporal train/test split utilities for recommendation evaluation.

Provides a simple per-student chronological holdout split: the newest
interactions per student are assigned to the test set according to
`test_fraction` while older interactions go to train.
"""
from typing import Tuple
import pandas as pd


def temporal_train_test_split(interactions_df: pd.DataFrame, test_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into train/test by time per student.

    Args:
        interactions_df: DataFrame with at least ['student_id', 'timestamp', 'topic'] columns.
        test_fraction: Fraction of each student's interactions to reserve for test (0..1).

    Returns:
        train_df, test_df (both DataFrames)
    """
    if 'timestamp' not in interactions_df.columns:
        # If no timestamp, fall back to existing ordering
        interactions_df = interactions_df.copy()
        interactions_df['_order'] = range(len(interactions_df))
        interactions_df = interactions_df.sort_values('_order')
    else:
        interactions_df = interactions_df.sort_values('timestamp')

    train_rows = []
    test_rows = []

    # Group by student and split chronologically
    for sid, group in interactions_df.groupby('student_id'):
        n = len(group)
        if n == 0:
            continue
        n_test = int(round(n * float(test_fraction)))
        # ensure at least 1 test if n>1 and fraction>0
        if n_test == 0 and n > 1 and test_fraction > 0.0:
            n_test = 1

        if n_test >= n:
            n_test = max(0, n - 1)

        if n_test > 0:
            test_part = group.tail(n_test)
            train_part = group.head(n - n_test)
        else:
            test_part = group.head(0)
            train_part = group

        train_rows.append(train_part)
        test_rows.append(test_part)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else interactions_df.head(0)
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else interactions_df.head(0)

    # Drop helper column if present
    if '_order' in train_df.columns:
        train_df = train_df.drop(columns=['_order'])
    if '_order' in test_df.columns:
        test_df = test_df.drop(columns=['_order'])

    return train_df, test_df
