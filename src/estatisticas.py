import csv
from collections import Counter

import numpy as np


def mode_solution(results, decimal_places=2):
    rounded_solutions = [tuple(np.round(result.x_best, decimal_places)) for result in results]
    solution, frequency = Counter(rounded_solutions).most_common(1)[0]
    return np.asarray(solution, dtype=float), frequency


def summarize_results(results_by_algorithm, target_value, decimal_places=2):
    summaries = []

    for algorithm, results in results_by_algorithm.items():
        values = np.asarray([result.f_best for result in results], dtype=float)
        iterations = np.asarray([result.iterations for result in results], dtype=float)
        best_index = int(np.argmin(values))
        best_result = results[best_index]
        mode_x, mode_frequency = mode_solution(results, decimal_places)
        success_count = int(np.sum(values <= target_value))

        summaries.append(
            {
                "algoritmo": algorithm,
                "rodadas": len(results),
                "hiperparametro": best_result.hyperparameter_name,
                "valor_hiperparametro": best_result.hyperparameter_value,
                "melhor_x1": best_result.x_best[0],
                "melhor_x2": best_result.x_best[1],
                "melhor_f": best_result.f_best,
                "media_f": float(np.mean(values)),
                "desvio_f": float(np.std(values)),
                "menor_f": float(np.min(values)),
                "maior_f": float(np.max(values)),
                "media_iteracoes": float(np.mean(iterations)),
                "moda_x1": mode_x[0],
                "moda_x2": mode_x[1],
                "frequencia_moda": mode_frequency,
                "sucessos": success_count,
                "taxa_sucesso": success_count / len(results),
            }
        )

    return summaries


def write_rounds_csv(results_by_algorithm, output_path, target_value):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algoritmo",
        "rodada",
        "hiperparametro",
        "valor_hiperparametro",
        "x1",
        "x2",
        "f",
        "iteracoes",
        "parada",
        "sucesso",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for algorithm, results in results_by_algorithm.items():
            for round_index, result in enumerate(results, start=1):
                writer.writerow(
                    {
                        "algoritmo": algorithm,
                        "rodada": round_index,
                        "hiperparametro": result.hyperparameter_name,
                        "valor_hiperparametro": _format_optional_float(result.hyperparameter_value),
                        "x1": _format_float(result.x_best[0]),
                        "x2": _format_float(result.x_best[1]),
                        "f": _format_float(result.f_best),
                        "iteracoes": result.iterations,
                        "parada": result.stopped_by,
                        "sucesso": int(result.f_best <= target_value),
                    }
                )


def write_summary_csv(summaries, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _format_value(value) for key, value in summary.items()})


def write_hyperparameter_csv(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _format_value(value) for key, value in record.items()})


def _format_value(value):
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, np.floating):
        return _format_float(float(value))
    return value


def _format_optional_float(value):
    if value is None:
        return ""
    return _format_float(value)


def _format_float(value):
    return f"{float(value):.10f}"
