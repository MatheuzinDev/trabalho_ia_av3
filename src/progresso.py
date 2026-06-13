import time

import numpy as np


def format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
    if minutes > 0:
        return f"{minutes:02d}m{seconds:02d}s"
    return f"{seconds:02d}s"


def calculate_eta(elapsed_seconds, completed, total):
    if completed <= 0:
        return 0.0

    average_seconds = elapsed_seconds / completed
    remaining = max(0, total - completed)
    return average_seconds * remaining


def should_report_progress(completed, total, interval):
    return completed == 1 or completed == total or completed % interval == 0


def build_partial_summary(results, target_value, minimize=True):
    values = np.asarray([result.f_best for result in results], dtype=float)
    iterations = np.asarray([result.iterations for result in results], dtype=float)

    if minimize:
        best_value = float(np.min(values))
        success_count = int(np.sum(values <= target_value))
    else:
        best_value = float(np.max(values))
        success_count = int(np.sum(values >= target_value))

    return {
        "best_value": best_value,
        "mean_value": float(np.mean(values)),
        "success_count": success_count,
        "success_rate": success_count / len(results),
        "mean_iterations": float(np.mean(iterations)),
    }


def print_run_header(problem, rounds, validation_rounds, max_iterations, random_seed):
    seed_label = "aleatoria" if random_seed is None else random_seed
    optimization_type = "minimizacao" if problem.minimize else "maximizacao"

    print(f"\n{problem.name}")
    print(f"Tipo de otimizacao: {optimization_type}")
    print(f"Valor alvo de sucesso: {problem.target_value}")
    print(f"Rodadas finais por algoritmo: {rounds}")
    print(f"Rodadas de validacao por hiperparametro: {validation_rounds}")
    print(f"Maximo de iteracoes por rodada: {max_iterations}")
    print(f"Patience: {problem.patience}")
    print(f"Seed utilizada: {seed_label}")


def print_phase_header(phase_index, phase_total, title):
    print(f"\n[{phase_index}/{phase_total}] {title}")


def print_validation_progress(parameter_name, parameter_value, record, completed, total, start_time):
    elapsed_seconds = time.perf_counter() - start_time
    eta_seconds = calculate_eta(elapsed_seconds, completed, total)
    success_rate = 100.0 * record["taxa_sucesso"]

    print(
        f"{parameter_name}={parameter_value:<8} | "
        f"{completed:>2}/{total} valores | "
        f"sucesso={record['sucessos']}/{record['rodadas_validacao']} ({success_rate:.2f}%) | "
        f"melhor_f={record['melhor_f']:.10f} | "
        f"media_f={record['media_f']:.10f} | "
        f"tempo={format_duration(elapsed_seconds)} | "
        f"ETA={format_duration(eta_seconds)}"
    )


def print_round_progress(algorithm, completed, total, results, target_value, minimize, start_time):
    elapsed_seconds = time.perf_counter() - start_time
    eta_seconds = calculate_eta(elapsed_seconds, completed, total)
    summary = build_partial_summary(results, target_value, minimize)

    print(
        f"{algorithm:<20} | "
        f"rodada {completed:>3}/{total} | "
        f"sucesso={summary['success_count']:>3} ({100.0 * summary['success_rate']:.2f}%) | "
        f"melhor_f={summary['best_value']:.10f} | "
        f"media_f={summary['mean_value']:.10f} | "
        f"iter_media={summary['mean_iterations']:.2f} | "
        f"tempo={format_duration(elapsed_seconds)} | "
        f"ETA={format_duration(eta_seconds)}"
    )


def print_file_step(message):
    print(f"- {message}")
