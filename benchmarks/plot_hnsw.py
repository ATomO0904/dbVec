import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/hnsw_results.csv")

summary = df.groupby(["backend", "ef_search"], as_index=False).agg(
    recall=("recall_at_k", "mean"),
    p95=("p95_ms", "mean"),
    qps=("qps", "mean"),
)

fig, ax = plt.subplots(figsize=(8, 5))

for backend, group in summary.groupby("backend"):
    group = group.sort_values("ef_search")

    ax.plot(
        group["p95"],
        group["recall"],
        marker="o",
        label=backend,
    )

    for _, row in group.iterrows():
        ax.annotate(
            f"ef={int(row['ef_search'])}",
            (row["p95"], row["recall"]),
            xytext=(5, 5),
            textcoords="offset points",
        )

ax.set_xlabel("Latencia p95 (ms)")
ax.set_ylabel("Recall@10")
ax.set_title("Qdrant vs pgvector - HNSW")
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

output = "results/hnsw_comparison.png"
fig.savefig(output, dpi=150)
print(f"Gráfico guardado en {output}")
