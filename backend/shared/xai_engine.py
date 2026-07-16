"""
xai_engine.py — shared pure-math primitives.

cosine_similarity is used by verification/edition_conflict.py. The
production compute paths for faithfulness, citation precision, related-query
lookup, and edition-conflict detection live in xai_matrices.py /
verification/trust_gate.py / verification/edition_conflict.py /
shared/database.py — not duplicated here.
"""

import math
from typing import List


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
    Returns value ∈ [-1, 1]. Returns 0.0 if either vector is zero.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
