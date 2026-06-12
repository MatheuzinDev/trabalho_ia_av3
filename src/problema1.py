import numpy as np


PROBLEM_NAME = "Problema 1 - Funcao quadratica"
BOUNDS = np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=float)
OPTIMUM_POINT = np.array([0.0, 0.0], dtype=float)
OPTIMUM_VALUE = 0.0
MINIMIZE = True


def objective_function(candidate):
    x = np.asarray(candidate, dtype=float)
    return float(np.sum(x**2))


def is_inside_bounds(candidate, bounds=BOUNDS):
    x = np.asarray(candidate, dtype=float)
    return bool(np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]))
