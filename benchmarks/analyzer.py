import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/insert_results.csv")

summary = df.groupby("backend")["vectors_per_second"].agg(["mean", "std"])
vectors = int(df["verified_count"].iloc[0])

fig, ax = plt.subplots(figsize=(7, 4))
summary["mean"].plot(
    kind="bar",
    ax=ax,
    yerr=summary["std"].fillna(0),
    capsize=5,
)

ax.set_ylabel("Vectores por segundo")
ax.set_title(f"Qdrant vs pgvector - Inserción masiva de {vectors:,} vectores")
ax.set_xlabel("Motor vectorial")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()

output = "results/insert_comparison.png"
fig.savefig(output, dpi=150)
print(f"Gráfico guardado en {output}")
