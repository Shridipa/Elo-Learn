import sys
import pathlib

# Ensure project root on path so imports work under pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from knowledge_graph.graph import create_sample_knowledge_graph


def test_infer_weaknesses_returns_ranked_list():
    kg = create_sample_knowledge_graph()
    # craft mastery with low mastery on 'Transformers' and some others
    mastery = {
        'Transformers': 0.2,
        'Backpropagation': 0.6,
        'Matrices': 0.4,
        'Linear Algebra Basics': 0.8,
    }

    weaknesses = kg.infer_weaknesses(mastery, low_mastery_threshold=0.5, propagation_depth=3)
    assert isinstance(weaknesses, list)
    # Expect at least one weakness suggested when Transformers is low
    assert len(weaknesses) > 0
    # Each entry should be (concept, score) and score in [0,1]
    for concept, score in weaknesses:
        assert isinstance(concept, str)
        assert 0.0 <= score <= 1.0


if __name__ == '__main__':
    test_infer_weaknesses_returns_ranked_list()
    print('OK')
