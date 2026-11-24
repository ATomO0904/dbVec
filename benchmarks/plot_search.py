import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/search_results.csv")

metrics = [
    ("qps", "Consultas por segundo"),
    ("p50_ms", "Latencia p50 (ms)"),
    ("p95_ms", "Latencia p95 (ms)"),
    ("p99_ms", "Latencia p99 (ms)"),
]

summary = df.groupby("backend").agg(
    {metric: ["mean", "std"] for metric, _ in metrics}
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (metric, ylabel) in zip(axes.flat, metrics):
    means = summary[(metric, "mean")]
    stds = summary[(metric, "std")].fillna(0)

    means.plot.bar(ax=ax, yerr=stds, capsize=5)

    ax.set_title(metric)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Motor vectorial")
    ax.tick_params(axis="x", rotation=0)

fig.suptitle("Qdrant vs pgvector - Búsqueda Top-k", fontsize=14)
fig.tight_layout()

output = "results/search_comparison.png"
fig.savefig(output, dpi=150)
print(f"Gráfico guardado en {output}")
