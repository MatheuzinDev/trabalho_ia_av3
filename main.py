from pathlib import Path

import numpy as np

from src.estatisticas import (
    count_successes,
    summarize_results,
    write_hyperparameter_csv,
    write_rounds_csv_with_direction,
    write_summary_csv,
)
from src.global_random_search import global_random_search
from src.graficos import save_convergence_plot, save_final_solutions_plot, save_final_solutions_zoom_plot
from src.hill_climbing import hill_climbing
from src.local_random_search import local_random_search
from src.problemas_continuos import get_problem


ROUNDS = 100
VALIDATION_ROUNDS = 100
MAX_ITERATIONS = 1000
PATIENCE = 200
MIN_VALIDATION_SUCCESS_RATE = 0.8
SOLUTION_DECIMAL_PLACES = 2
# use None para execuções aleatórios ou use um inteiro para reproduzir sempre os mesmos resultados
RANDOM_SEED = None
SELECTED_PROBLEM_ID = "problema3"

EPSILON_VALUES = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0)
SIGMA_VALUES = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5)


def main():
    project_dir = Path(__file__).resolve().parent
    problem = get_problem(SELECTED_PROBLEM_ID)
    output_dir = project_dir / "resultados" / problem.problem_id
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_records, selected_epsilon = select_hill_climbing_epsilon(problem)
    sigma_records, selected_sigma = select_lrs_sigma(problem)
    write_hyperparameter_csv(epsilon_records + sigma_records, output_dir / "hiperparametros.csv")

    rng = create_rng(RANDOM_SEED)
    results_by_algorithm = {
        "Hill Climbing": run_hill_climbing_rounds(problem, selected_epsilon, rng),
        "Local Random Search": run_lrs_rounds(problem, selected_sigma, rng),
        "Global Random Search": run_grs_rounds(problem, rng),
    }

    summaries = summarize_results(
        results_by_algorithm,
        problem.target_value,
        problem.minimize,
        SOLUTION_DECIMAL_PLACES,
    )
    write_rounds_csv_with_direction(
        results_by_algorithm,
        output_dir / "rodadas.csv",
        problem.target_value,
        problem.minimize,
    )
    write_summary_csv(summaries, output_dir / "resumo.csv")
    save_convergence_plot(results_by_algorithm, output_dir / "convergencia.png", problem.name)
    save_final_solutions_plot(
        results_by_algorithm,
        problem.optimum_point,
        problem.bounds,
        output_dir / "solucoes_finais.png",
        problem.name,
    )
    save_final_solutions_zoom_plot(
        results_by_algorithm,
        problem.optimum_point,
        output_dir / "solucoes_finais_zoom.png",
        problem.zoom_limit,
        problem.name,
    )

    print_execution_summary(problem, summaries, selected_epsilon, selected_sigma, output_dir, RANDOM_SEED)


def create_rng(seed=None, offset=0):
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed + offset)


def select_hill_climbing_epsilon(problem):
    records = []

    for value_index, epsilon in enumerate(EPSILON_VALUES):
        rng = create_rng(RANDOM_SEED, 1000 + value_index)
        results = run_hill_climbing_rounds(problem, epsilon, rng, rounds=VALIDATION_ROUNDS)
        records.append(build_hyperparameter_record(problem, "Hill Climbing", "epsilon", epsilon, results))

    return records, select_smallest_successful_value(records, problem.minimize)


def select_lrs_sigma(problem):
    records = []

    for value_index, sigma in enumerate(SIGMA_VALUES):
        rng = create_rng(RANDOM_SEED, 2000 + value_index)
        results = run_lrs_rounds(problem, sigma, rng, rounds=VALIDATION_ROUNDS)
        records.append(build_hyperparameter_record(problem, "Local Random Search", "sigma", sigma, results))

    return records, select_smallest_successful_value(records, problem.minimize)


def run_hill_climbing_rounds(problem, epsilon, rng, rounds=ROUNDS):
    return [
        hill_climbing(
            problem.objective_function,
            problem.bounds,
            epsilon,
            MAX_ITERATIONS,
            problem.patience,
            problem.target_value,
            rng,
            problem.minimize,
        )
        for _ in range(rounds)
    ]


def run_lrs_rounds(problem, sigma, rng, rounds=ROUNDS):
    return [
        local_random_search(
            problem.objective_function,
            problem.bounds,
            sigma,
            MAX_ITERATIONS,
            problem.patience,
            problem.target_value,
            rng,
            problem.minimize,
        )
        for _ in range(rounds)
    ]


def run_grs_rounds(problem, rng, rounds=ROUNDS):
    return [
        global_random_search(
            problem.objective_function,
            problem.bounds,
            MAX_ITERATIONS,
            problem.patience,
            problem.target_value,
            rng,
            problem.minimize,
        )
        for _ in range(rounds)
    ]


def build_hyperparameter_record(problem, algorithm, parameter_name, parameter_value, results):
    values = np.asarray([result.f_best for result in results], dtype=float)
    success_count = count_successes(values, problem.target_value, problem.minimize)
    best_value = float(np.min(values) if problem.minimize else np.max(values))
    worst_value = float(np.max(values) if problem.minimize else np.min(values))

    return {
        "algoritmo": algorithm,
        "parametro": parameter_name,
        "valor": parameter_value,
        "rodadas_validacao": len(results),
        "sucessos": success_count,
        "taxa_sucesso": success_count / len(results),
        "media_f": float(np.mean(values)),
        "mediana_f": float(np.median(values)),
        "melhor_f": best_value,
        "pior_f": worst_value,
    }


def select_smallest_successful_value(records, minimize=True):
    successful_records = [
        record
        for record in records
        if record["taxa_sucesso"] >= MIN_VALIDATION_SUCCESS_RATE
    ]
    if successful_records:
        return successful_records[0]["valor"]

    if minimize:
        best_record = min(
            records,
            key=lambda record: (-record["taxa_sucesso"], record["mediana_f"], record["media_f"]),
        )
    else:
        best_record = min(
            records,
            key=lambda record: (-record["taxa_sucesso"], -record["mediana_f"], -record["media_f"]),
        )
    return best_record["valor"]


def print_execution_summary(problem, summaries, selected_epsilon, selected_sigma, output_dir, random_seed):
    print(f"\n{problem.name} finalizado")
    seed_label = "aleatoria" if random_seed is None else random_seed
    print(f"Seed utilizada: {seed_label}")
    print(f"Tipo de otimizacao: {'minimizacao' if problem.minimize else 'maximizacao'}")
    print(f"Valor alvo de sucesso: {problem.target_value}")
    print(f"Patience: {problem.patience}")
    print(f"Epsilon selecionado para Hill Climbing: {selected_epsilon}")
    print(f"Sigma selecionado para LRS: {selected_sigma}")
    print(f"Arquivos gerados em: {output_dir}")

    for summary in summaries:
        print(
            "\n"
            f"{summary['algoritmo']}\n"
            f"  melhor x = ({summary['melhor_x1']:.6f}, {summary['melhor_x2']:.6f})\n"
            f"  melhor f(x) = {summary['melhor_f']:.10f}\n"
            f"  media f(x) = {summary['media_f']:.10f}\n"
            f"  moda x = ({summary['moda_x1']:.2f}, {summary['moda_x2']:.2f}) "
            f"com frequencia {summary['frequencia_moda']}\n"
            f"  taxa de sucesso = {summary['taxa_sucesso']:.2%}\n"
            f"  iteracoes medias = {summary['media_iteracoes']:.2f}"
        )


if __name__ == "__main__":
    main()
