import numpy as np


class SearchResult:
    def __init__(
        self,
        algorithm,
        x_best,
        f_best,
        iterations,
        stopped_by,
        convergence,
        hyperparameter_name="",
        hyperparameter_value=None,
    ):
        self.algorithm = algorithm
        self.x_best = x_best
        self.f_best = f_best
        self.iterations = iterations
        self.stopped_by = stopped_by
        self.convergence = convergence
        self.hyperparameter_name = hyperparameter_name
        self.hyperparameter_value = hyperparameter_value


def is_in_bounds(candidate, bounds):
    x = np.asarray(candidate, dtype=float)
    return bool(np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]))


def is_better(candidate_value, best_value, minimize=True):
    if minimize:
        return candidate_value < best_value
    return candidate_value > best_value


def target_reached(value, target_value, minimize=True):
    if target_value is None:
        return False
    if minimize:
        return value <= target_value
    return value >= target_value


def sample_uniform_neighborhood(center, epsilon, bounds, rng, max_attempts=100):
    center = np.asarray(center, dtype=float)
    lower_limits = bounds[:, 0]
    upper_limits = bounds[:, 1]

    candidate = center.copy()
    for _ in range(max_attempts):
        candidate = center + rng.uniform(-epsilon, epsilon, size=center.shape)
        if is_in_bounds(candidate, bounds):
            return candidate

    return np.clip(candidate, lower_limits, upper_limits)


def sample_gaussian_neighborhood(center, sigma, bounds, rng, max_attempts=100):
    center = np.asarray(center, dtype=float)
    lower_limits = bounds[:, 0]
    upper_limits = bounds[:, 1]
    domain_range = upper_limits - lower_limits
    scale = sigma * domain_range

    candidate = center.copy()
    for _ in range(max_attempts):
        candidate = center + rng.normal(0.0, scale, size=center.shape)
        if is_in_bounds(candidate, bounds):
            return candidate

    return np.clip(candidate, lower_limits, upper_limits)
