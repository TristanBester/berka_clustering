from src.modules.clustering.metrics import (
    complexity_invariant_similarity,
    correlation_based_similarity,
    euclidean_distance,
)


def metric_factory(metric):
    if metric == "eucl":
        return euclidean_distance
    if metric == "corr":
        return correlation_based_similarity
    if metric == "cid":
        return complexity_invariant_similarity
