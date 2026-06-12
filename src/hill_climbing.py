from src.busca_utils import SearchResult, is_better, sample_uniform_neighborhood, target_reached


def hill_climbing(
    objective_function,
    bounds,
    epsilon,
    max_iterations,
    patience,
    target_value,
    rng,
    minimize=True,
):
    x_best = bounds[:, 0].astype(float).copy()
    f_best = objective_function(x_best)
    convergence = [f_best]
    iterations_without_improvement = 0
    stopped_by = "max_iterations"
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        candidate = sample_uniform_neighborhood(x_best, epsilon, bounds, rng)
        candidate_value = objective_function(candidate)

        if is_better(candidate_value, f_best, minimize):
            x_best = candidate
            f_best = candidate_value
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        convergence.append(f_best)

        if target_reached(f_best, target_value, minimize):
            stopped_by = "target_value"
            break

        if iterations_without_improvement >= patience:
            stopped_by = "patience"
            break

    return SearchResult(
        algorithm="Hill Climbing",
        x_best=x_best,
        f_best=f_best,
        iterations=iteration,
        stopped_by=stopped_by,
        convergence=convergence,
        hyperparameter_name="epsilon",
        hyperparameter_value=epsilon,
    )
