"""Regenera results/resumen_benchmarks.png a partir de los CSV de results/.

Uso: python -m benchmarks.plot_summary
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
COLORS = {"pgvector": "#4C72B0", "qdrant": "#DD8452"}
ORDER = ["pgvector", "qdrant"]


def bars(ax, values, errors, title, ylabel, log=False):
    positions = range(len(ORDER))
    ax.bar(
        positions,
        [values.get(name, 0) for name in ORDER],
        yerr=[errors.get(name, 0) for name in ORDER],
        color=[COLORS[name] for name in ORDER],
        capsize=5,
    )
    ax.set_xticks(list(positions), ORDER)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if log:
        ax.set_yscale("log")


def main():
    insert = pd.read_csv(RESULTS / "insert_results.csv").groupby("backend")[
        "vectors_per_second"
    ].mean()
    search = pd.read_csv(RESULTS / "search_results.csv").groupby("backend")["qps"].agg(
        ["mean", "std"]
    )
    hnsw = pd.read_csv(RESULTS / "hnsw_results.csv").groupby(
        ["backend", "ef_search"], as_index=False
    ).agg(recall=("recall_at_k", "mean"), p95=("p95_ms", "mean"))
    memory = pd.read_csv(RESULTS / "memory_results.csv").groupby("backend")[
        "delta_peak_rss_mb"
    ].mean()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    bars(
        axes[0, 0], insert.to_dict(), {},
        "Inserción (1 M vectores)", "Vectores/s", log=True,
    )
    for i, name in enumerate(ORDER):
        axes[0, 0].text(
            i, insert[name], f" {insert[name]:,.0f}", ha="center",
            va="bottom", fontsize=9,
        )

    bars(
        axes[0, 1], search["mean"].to_dict(), search["std"].fillna(0).to_dict(),
        "Búsqueda Top-k (10 k vectores)", "QPS (media ± sd)",
    )
    for i, name in enumerate(ORDER):
        axes[0, 1].text(
            i, search.loc[name, "mean"], f" {search.loc[name, 'mean']:.0f}",
            ha="center", va="bottom", fontsize=9,
        )

    for backend, group in hnsw.groupby("backend"):
        group = group.sort_values("ef_search")
        axes[1, 0].plot(
            group["p95"], group["recall"], marker="o", label=backend,
            color=COLORS[backend],
        )
        for _, row in group.iterrows():
            axes[1, 0].annotate(
                f"ef={int(row['ef_search'])}", (row["p95"], row["recall"]),
                xytext=(4, 5), textcoords="offset points", fontsize=8,
            )
    axes[1, 0].set_title("HNSW semántico (1 k vectores)")
    axes[1, 0].set_xlabel("Latencia p95 (ms)")
    axes[1, 0].set_ylabel("Recall@10")
    axes[1, 0].set_ylim(0.95, 1.005)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    bars(
        axes[1, 1], memory.to_dict(), {},
        "RAM (100 k vectores)", "Incremento residente pico (MiB)",
    )
    for i, name in enumerate(ORDER):
        axes[1, 1].text(
            i, memory[name], f" {memory[name]:.0f} MiB",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle("pgvector vs Qdrant — resumen de benchmarks", fontsize=15)
    fig.tight_layout()
    output = RESULTS / "resumen_benchmarks.png"
    fig.savefig(output, dpi=160)
    print(f"Escrito: {output}")


if __name__ == "__main__":
    main()
