from src.modules.clustering.metrics import (
    complexity_invariant_similarity_np, complexity_invariant_similarity_torch,
    correlation_based_similarity_np, correlation_based_similarity_torch,
    euclidean_distance_np, euclidean_distance_torch)


def metric_factory(metric):
    if metric == "eucl":
        return euclidean_distance_np, euclidean_distance_torch
    if metric == "corr":
        return correlation_based_similarity_np, correlation_based_similarity_torch
    if metric == "cid":
        return complexity_invariant_similarity_np, complexity_invariant_similarity_torch
