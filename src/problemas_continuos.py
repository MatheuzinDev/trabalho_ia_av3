import numpy as np


class ContinuousProblem:
    def __init__(
        self,
        problem_id,
        name,
        objective_function,
        bounds,
        minimize,
        target_value,
        optimum_point,
        optimum_value,
        zoom_limit,
        patience,
        epsilon_values,
        sigma_values,
    ):
        self.problem_id = problem_id
        self.name = name
        self.objective_function = objective_function
        self.bounds = np.asarray(bounds, dtype=float)
        self.minimize = minimize
        self.target_value = target_value
        self.optimum_point = np.asarray(optimum_point, dtype=float)
        self.optimum_value = optimum_value
        self.zoom_limit = zoom_limit
        self.patience = patience
        self.epsilon_values = tuple(float(value) for value in epsilon_values)
        self.sigma_values = tuple(float(value) for value in sigma_values)


def objective_problem_1(candidate):
    x = np.asarray(candidate, dtype=float)
    return float(np.sum(x**2))


def objective_problem_2(candidate):
    x = np.asarray(candidate, dtype=float)
    x1 = x[0]
    x2 = x[1]
    first_term = np.exp(-(x1**2 + x2**2))
    second_term = 2.0 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2))
    return float(first_term + second_term)


def objective_problem_3(candidate):
    x = np.asarray(candidate, dtype=float)
    x1 = x[0]
    x2 = x[1]
    first_term = -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
    second_term = -np.exp(0.5 * (np.cos(2.0 * np.pi * x1) + np.cos(2.0 * np.pi * x2)))
    return float(first_term + second_term + 20.0 + np.e)


def objective_problem_4(candidate):
    x = np.asarray(candidate, dtype=float)
    x1 = x[0]
    x2 = x[1]
    first_term = x1**2 - 10.0 * np.cos(2.0 * np.pi * x1) + 10.0
    second_term = x2**2 - 10.0 * np.cos(2.0 * np.pi * x2) + 10.0
    return float(first_term + second_term)


def objective_problem_5(candidate):
    x = np.asarray(candidate, dtype=float)
    x1 = x[0]
    x2 = x[1]
    first_term = (x1 * np.cos(x1)) / 20.0
    second_term = 2.0 * np.exp(-(x1**2) - ((x2 - 1.0) ** 2))
    third_term = 0.01 * x1 * x2
    return float(first_term + second_term + third_term)


PROBLEMS = {
    "problema1": ContinuousProblem(
        problem_id="problema1",
        name="Problema 1 - Funcao quadratica",
        objective_function=objective_problem_1,
        bounds=np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=float),
        minimize=True,
        target_value=1e-2,
        optimum_point=np.array([0.0, 0.0], dtype=float),
        optimum_value=0.0,
        zoom_limit=1.0,
        patience=200,
        epsilon_values=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
        sigma_values=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1),
    ),
    "problema2": ContinuousProblem(
        problem_id="problema2",
        name="Problema 2 - Soma de gaussianas",
        objective_function=objective_problem_2,
        bounds=np.array([[-2.0, 4.0], [-2.0, 5.0]], dtype=float),
        minimize=False,
        target_value=2.0,
        optimum_point=np.array([1.7, 1.7], dtype=float),
        optimum_value=2.0,
        zoom_limit=1.0,
        patience=200,
        epsilon_values=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
        sigma_values=(0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1),
    ),
    "problema3": ContinuousProblem(
        problem_id="problema3",
        name="Problema 3 - Funcao Ackley",
        objective_function=objective_problem_3,
        bounds=np.array([[-8.0, 8.0], [-8.0, 8.0]], dtype=float),
        minimize=True,
        target_value=1e-1,
        optimum_point=np.array([0.0, 0.0], dtype=float),
        optimum_value=0.0,
        zoom_limit=1.0,
        patience=1000,
        epsilon_values=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0),
        sigma_values=(0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1),
    ),
    "problema4": ContinuousProblem(
        problem_id="problema4",
        name="Problema 4 - Funcao Rastrigin",
        objective_function=objective_problem_4,
        bounds=np.array([[-5.12, 5.12], [-5.12, 5.12]], dtype=float),
        minimize=True,
        target_value=1e-1,
        optimum_point=np.array([0.0, 0.0], dtype=float),
        optimum_value=0.0,
        zoom_limit=1.0,
        patience=1000,
        epsilon_values=(0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 5.0),
        sigma_values=(0.025, 0.03, 0.04, 0.045, 0.05, 0.055, 0.06, 0.075, 0.1, 0.15, 0.25),
    ),
    "problema5": ContinuousProblem(
        problem_id="problema5",
        name="Problema 5 - Funcao exponencial com termo oscilatorio",
        objective_function=objective_problem_5,
        bounds=np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=float),
        minimize=False,
        target_value=2.0,
        optimum_point=np.array([0.015, 1.00004], dtype=float),
        optimum_value=2.000449969049,
        zoom_limit=1.0,
        patience=1000,
        epsilon_values=(0.1, 1.0, 3.0, 5.0, 10.0, 12.0, 15.0, 18.0, 20.0),
        sigma_values=(0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.04, 0.05),
    ),
}


def get_problem(problem_id):
    try:
        return PROBLEMS[problem_id]
    except KeyError as exc:
        available = ", ".join(sorted(PROBLEMS))
        raise ValueError(f"Problema desconhecido: {problem_id}. Disponiveis: {available}") from exc
