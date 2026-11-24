import argparse
import asyncio
import csv
import time
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select, func

from db.pgvector_db import session_local, engine
from models.models import Document, DIMENSIONS
from repository.pgvector_repository import PgVectorRepository
from repository.qdrant_repository import (
    QdrantRepository,
    COLLECTION,
)


def generate_queries(
    count: int,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)

    queries = rng.standard_normal(
        (count, DIMENSIONS),
        dtype=np.float32,
    )

    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    return queries


def percentile(values, p):
    return float(np.percentile(values, p))


async def benchmark_search(
    repository,
    queries,
    k: int,
    warmup: int,
):
    latencies = []
    rankings = []

    # Calentamiento
    for vector in queries[:warmup]:
        await repository.search(vector.tolist(), k)

    # Medición
    start_total = time.perf_counter()

    for vector in queries:
        start = time.perf_counter()

        results = await repository.search(
            vector.tolist(),
            k,
        )

        elapsed = time.perf_counter() - start

        latencies.append(elapsed * 1000)
        rankings.append([r["id"] for r in results])

    total_time = time.perf_counter() - start_total

    return {
        "latencies": latencies,
        "rankings": rankings,
        "total_seconds": total_time,
        "qps": len(queries) / total_time,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": float(np.mean(latencies)),
    }


async def main(args):
    queries = generate_queries(args.queries)

    client = AsyncQdrantClient(
        url="http://localhost:6333",
        timeout=120,
    )

    rows = []

    try:
        async with session_local() as session:
            pg = PgVectorRepository(session)
            qdrant = QdrantRepository(client)

            pg_count = (
                await session.execute(select(func.count()).select_from(Document))
            ).scalar_one()

            qdrant_count = (
                await client.count(
                    collection_name=COLLECTION,
                    exact=True,
                )
            ).count

            if pg_count != args.size or qdrant_count != args.size:
                raise RuntimeError(
                    f"Se esperaban {args.size} vectores. "
                    f"pgvector={pg_count}, Qdrant={qdrant_count}"
                )

            repositories = {
                "pgvector": pg,
                "qdrant": qdrant,
            }

            for run in range(args.runs):
                order = (
                    ["qdrant", "pgvector"] if run % 2 == 0 else ["pgvector", "qdrant"]
                )

                run_results = {}

                for backend in order:
                    print(f"Run {run + 1}: {backend}")

                    result = await benchmark_search(
                        repository=repositories[backend],
                        queries=queries,
                        k=args.k,
                        warmup=min(args.warmup, len(queries)),
                    )

                    run_results[backend] = result

                    rows.append(
                        {
                            "run": run + 1,
                            "backend": backend,
                            "vectors": args.size,
                            "queries": args.queries,
                            "k": args.k,
                            "qps": round(result["qps"], 2),
                            "mean_ms": round(result["mean_ms"], 4),
                            "p50_ms": round(result["p50_ms"], 4),
                            "p95_ms": round(result["p95_ms"], 4),
                            "p99_ms": round(result["p99_ms"], 4),
                        }
                    )

                # Comparar conjuntos de vecinos
                matches = []

                for pg_ids, q_ids in zip(
                    run_results["pgvector"]["rankings"],
                    run_results["qdrant"]["rankings"],
                ):
                    matches.append(len(set(pg_ids) & set(q_ids)) / args.k)

                agreement = np.mean(matches)

                print(f"Coincidencia Top-{args.k}: {agreement:.4%}\n")

    finally:
        await client.close()
        await engine.dispose()

    output = Path("results/search_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)

    args = parser.parse_args()

    if (
        args.size <= 0
        or args.queries <= 0
        or args.k <= 0
        or args.k > args.size
        or args.warmup < 0
        or args.runs <= 0
    ):
        parser.error("Parámetros de benchmark inválidos")

    asyncio.run(main(args))
