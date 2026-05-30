import pandas as pd
import datetime

from recommendation_engine.temporal_split import temporal_train_test_split


def make_interactions(student_id, n, start_ts):
    rows = []
    for i in range(n):
        rows.append({'student_id': student_id, 'topic': f't{ i }', 'timestamp': start_ts + datetime.timedelta(days=i)})
    return rows


def test_temporal_split_ordering():
    # Create interactions for two students with timestamps
    start = datetime.datetime(2020, 1, 1)
    rows = []
    rows += make_interactions(1, 5, start)
    rows += make_interactions(2, 3, start)

    df = pd.DataFrame(rows)

    train, test = temporal_train_test_split(df, test_fraction=0.4)

    # Student 1: 5 interactions -> round(5*0.4)=2 test -> last 2 in test
    s1_train = train[train['student_id'] == 1].sort_values('timestamp')
    s1_test = test[test['student_id'] == 1].sort_values('timestamp')
    assert len(s1_train) == 3
    assert len(s1_test) == 2
    # ensure chronological order: last test timestamp > last train timestamp
    assert s1_test['timestamp'].min() > s1_train['timestamp'].max()

    # Student 2: 3 interactions -> round(3*0.4)=1 test
    s2_train = train[train['student_id'] == 2].sort_values('timestamp')
    s2_test = test[test['student_id'] == 2].sort_values('timestamp')
    assert len(s2_train) == 2
    assert len(s2_test) == 1
    assert s2_test['timestamp'].min() > s2_train['timestamp'].max()


def test_temporal_split_no_timestamp():
    # If no timestamp, ensure ordering fallback does not crash
    rows = [
        {'student_id': 1, 'topic': 'a'},
        {'student_id': 1, 'topic': 'b'},
        {'student_id': 1, 'topic': 'c'},
    ]
    df = pd.DataFrame(rows)
    train, test = temporal_train_test_split(df, test_fraction=0.34)
    # Expect at least one test for n>1
    assert len(train) + len(test) == 3
    assert len(test) >= 1
