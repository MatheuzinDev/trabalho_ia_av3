import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALGORITHM_COLORS = {
    "Hill Climbing": "tab:blue",
    "Local Random Search": "tab:orange",
    "Global Random Search": "tab:green",
}


def save_convergence_plot(results_by_algorithm, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(9, 6))

    for algorithm, results in results_by_algorithm.items():
        mean_curve = mean_convergence_curve(results)
        ax.plot(
            range(len(mean_curve)),
            np.maximum(mean_curve, 1e-12),
            color=ALGORITHM_COLORS.get(algorithm),
            label=algorithm,
        )

    ax.set_title("Convergencia media - Problema 1")
    ax.set_xlabel("Iteracao")
    ax.set_ylabel("Melhor f(x) medio")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_final_solutions_plot(results_by_algorithm, optimum_point, bounds, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(7, 7))

    for algorithm, results in results_by_algorithm.items():
        points = np.asarray([result.x_best for result in results], dtype=float)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            alpha=0.65,
            color=ALGORITHM_COLORS.get(algorithm),
            label=algorithm,
        )

    ax.scatter(
        [optimum_point[0]],
        [optimum_point[1]],
        color="black",
        marker="x",
        s=90,
        linewidths=2,
        label="Otimo conhecido",
    )
    ax.set_title("Solucoes finais - Problema 1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.grid(alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_final_solutions_zoom_plot(results_by_algorithm, optimum_point, output_path, zoom_limit=1.0):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(7, 7))
    plot_order = ("Global Random Search", "Hill Climbing", "Local Random Search")

    for zorder, algorithm in enumerate(plot_order, start=1):
        results = results_by_algorithm.get(algorithm)
        if results is None:
            continue

        points = np.asarray([result.x_best for result in results], dtype=float)
        ax.scatter(
            points[:, 0],
            points[:, 1],
            alpha=0.7,
            s=25,
            color=ALGORITHM_COLORS.get(algorithm),
            label=algorithm,
            zorder=zorder,
        )

    ax.scatter(
        [optimum_point[0]],
        [optimum_point[1]],
        color="black",
        marker="x",
        s=80,
        linewidths=2,
        label="Otimo conhecido",
        zorder=4,
    )
    ax.set_title("Solucoes finais com zoom - Problema 1")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_xlim(-zoom_limit, zoom_limit)
    ax.set_ylim(-zoom_limit, zoom_limit)
    ax.grid(alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def mean_convergence_curve(results):
    max_length = max(len(result.convergence) for result in results)
    curves = np.zeros((len(results), max_length), dtype=float)

    for row_index, result in enumerate(results):
        curve = np.asarray(result.convergence, dtype=float)
        curves[row_index, : len(curve)] = curve
        curves[row_index, len(curve) :] = curve[-1]

    return np.mean(curves, axis=0)
