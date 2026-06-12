from pathlib import Path

import numpy as np

from src.estatisticas import summarize_results, write_hyperparameter_csv, write_rounds_csv, write_summary_csv
from src.global_random_search import global_random_search
from src.graficos import save_convergence_plot, save_final_solutions_plot
from src.hill_climbing import hill_climbing
from src.local_random_search import local_random_search
from src.problema1 import BOUNDS, MINIMIZE, OPTIMUM_POINT, objective_function


ROUNDS = 100
VALIDATION_ROUNDS = 20
MAX_ITERATIONS = 1000
PATIENCE = 200
TARGET_VALUE = 1e-2
MIN_VALIDATION_SUCCESS_RATE = 0.8
SOLUTION_DECIMAL_PLACES = 2
RANDOM_SEED = 42

EPSILON_VALUES = (0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)
SIGMA_VALUES = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "resultados" / "problema1"
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilon_records, selected_epsilon = select_hill_climbing_epsilon()
    sigma_records, selected_sigma = select_lrs_sigma()
    write_hyperparameter_csv(epsilon_records + sigma_records, output_dir / "hiperparametros.csv")

    rng = np.random.default_rng(RANDOM_SEED)
    results_by_algorithm = {
        "Hill Climbing": run_hill_climbing_rounds(selected_epsilon, rng),
        "Local Random Search": run_lrs_rounds(selected_sigma, rng),
        "Global Random Search": run_grs_rounds(rng),
    }

    summaries = summarize_results(results_by_algorithm, TARGET_VALUE, SOLUTION_DECIMAL_PLACES)
    write_rounds_csv(results_by_algorithm, output_dir / "rodadas.csv", TARGET_VALUE)
    write_summary_csv(summaries, output_dir / "resumo.csv")
    save_convergence_plot(results_by_algorithm, output_dir / "convergencia.png")
    save_final_solutions_plot(results_by_algorithm, OPTIMUM_POINT, BOUNDS, output_dir / "solucoes_finais.png")

    print_execution_summary(summaries, selected_epsilon, selected_sigma, output_dir)


def select_hill_climbing_epsilon():
    records = []

    for value_index, epsilon in enumerate(EPSILON_VALUES):
        rng = np.random.default_rng(RANDOM_SEED + 1000 + value_index)
        results = run_hill_climbing_rounds(epsilon, rng, rounds=VALIDATION_ROUNDS)
        records.append(build_hyperparameter_record("Hill Climbing", "epsilon", epsilon, results))

    return records, select_smallest_successful_value(records)


def select_lrs_sigma():
    records = []

    for value_index, sigma in enumerate(SIGMA_VALUES):
        rng = np.random.default_rng(RANDOM_SEED + 2000 + value_index)
        results = run_lrs_rounds(sigma, rng, rounds=VALIDATION_ROUNDS)
        records.append(build_hyperparameter_record("Local Random Search", "sigma", sigma, results))

    return records, select_smallest_successful_value(records)


def run_hill_climbing_rounds(epsilon, rng, rounds=ROUNDS):
    return [
        hill_climbing(
            objective_function,
            BOUNDS,
            epsilon,
            MAX_ITERATIONS,
            PATIENCE,
            TARGET_VALUE,
            rng,
            MINIMIZE,
        )
        for _ in range(rounds)
    ]


def run_lrs_rounds(sigma, rng, rounds=ROUNDS):
    return [
        local_random_search(
            objective_function,
            BOUNDS,
            sigma,
            MAX_ITERATIONS,
            PATIENCE,
            TARGET_VALUE,
            rng,
            MINIMIZE,
        )
        for _ in range(rounds)
    ]


def run_grs_rounds(rng, rounds=ROUNDS):
    return [
        global_random_search(
            objective_function,
            BOUNDS,
            MAX_ITERATIONS,
            PATIENCE,
            TARGET_VALUE,
            rng,
            MINIMIZE,
        )
        for _ in range(rounds)
    ]


def build_hyperparameter_record(algorithm, parameter_name, parameter_value, results):
    values = np.asarray([result.f_best for result in results], dtype=float)
    success_count = int(np.sum(values <= TARGET_VALUE))

    return {
        "algoritmo": algorithm,
        "parametro": parameter_name,
        "valor": parameter_value,
        "rodadas_validacao": len(results),
        "sucessos": success_count,
        "taxa_sucesso": success_count / len(results),
        "media_f": float(np.mean(values)),
        "mediana_f": float(np.median(values)),
        "melhor_f": float(np.min(values)),
        "pior_f": float(np.max(values)),
    }


def select_smallest_successful_value(records):
    successful_records = [
        record
        for record in records
        if record["taxa_sucesso"] >= MIN_VALIDATION_SUCCESS_RATE
    ]
    if successful_records:
        return successful_records[0]["valor"]

    best_record = min(
        records,
        key=lambda record: (-record["taxa_sucesso"], record["mediana_f"], record["media_f"]),
    )
    return best_record["valor"]


def print_execution_summary(summaries, selected_epsilon, selected_sigma, output_dir):
    print("\nProblema 1 finalizado")
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
