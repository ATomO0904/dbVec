import argparse
import asyncio
import csv
import time
from pathlib import Path

import numpy as np
from sqlalchemy import select, text, func
from qdrant_client import AsyncQdrantClient

from db.pgvector_db import session_local, engine
from models.models import Document, DIMENSIONS
from repository.pgvector_repository import PgVectorRepository
from repository.qdrant_repository import (
    QdrantRepository,
    COLLECTION,
)


def generate_queries(count, seed=2026):
    rng = np.random.default_rng(seed)

    vectors = rng.standard_normal(
        (count, DIMENSIONS),
        dtype=np.float32,
    )

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors.tolist()


async def exact_search(session, vector, k):
    # Evitar que PostgreSQL utilice el índice ANN.
    await session.execute(text("SET LOCAL enable_indexscan = off"))
    await session.execute(text("SET LOCAL enable_bitmapscan = off"))

    distance = Document.embedding.cosine_distance(vector)

    result = await session.execute(select(Document.id).order_by(distance).limit(k))

    ids = result.scalars().all()

    # Finalizar transacción y restaurar configuración local
    await session.rollback()

    return ids


def recall_at_k(actual, expected, k):
    return len(set(actual) & set(expected)) / k


async def evaluate(
    repository,
    backend,
    queries,
    ground_truth,
    k,
    ef,
    warmup,
):
    latencies = []
    recalls = []

    # Warmup: consultas sin medición
    for vector in queries[:warmup]:
        await repository.search_hnsw(vector, limit=k, ef_search=ef)

        if backend == "pgvector":
            await repository.session.rollback()

    start_total = time.perf_counter()

    for vector, expected in zip(queries, ground_truth):
        start = time.perf_counter()

        results = await repository.search_hnsw(vector, limit=k, ef_search=ef)

        # Para pgvector, cerrar la transacción después
        # de recibir los resultados.
        if backend == "pgvector":
            await repository.session.rollback()

        elapsed = time.perf_counter() - start

        actual = [row["id"] for row in results]

        latencies.append(elapsed * 1000)
        recalls.append(recall_at_k(actual, expected, k))

    total = time.perf_counter() - start_total

    return {
        "backend": backend,
        "ef_search": ef,
        "recall_at_k": float(np.mean(recalls)),
        "qps": len(queries) / total,
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
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
            pg_count = (
                await session.execute(select(func.count()).select_from(Document))
            ).scalar_one()
            await session.rollback()

            q_count = (
                await client.count(
                    collection_name=COLLECTION,
                    exact=True,
                )
            ).count

            if pg_count != args.size or q_count != args.size:
                raise RuntimeError(
                    f"Dataset incorrecto: pgvector={pg_count}, Qdrant={q_count}"
                )

            pg = PgVectorRepository(session)
            qdrant = QdrantRepository(client)

            print("Calculando ground truth exacto...")

            ground_truth = []

            for i, vector in enumerate(queries):
                ids = await exact_search(session, vector, args.k)
                ground_truth.append(ids)

                if (i + 1) % 10 == 0:
                    print(f"Exactas: {i + 1}/{len(queries)}")

            repositories = {
                "pgvector": pg,
                "qdrant": qdrant,
            }

            for run in range(args.runs):
                for ef in args.ef_values:
                    order = (
                        ["qdrant", "pgvector"]
                        if run % 2 == 0
                        else ["pgvector", "qdrant"]
                    )

                    for backend in order:
                        print(f"Run={run + 1} backend={backend} ef={ef}")

                        result = await evaluate(
                            repository=repositories[backend],
                            backend=backend,
                            queries=queries,
                            ground_truth=ground_truth,
                            k=args.k,
                            ef=ef,
                            warmup=min(args.warmup, len(queries)),
                        )

                        result.update(
                            {
                                "run": run + 1,
                                "vectors": args.size,
                                "queries": args.queries,
                                "k": args.k,
                            }
                        )

                        rows.append(result)
                        print(result)

    finally:
        await client.close()
        await engine.dispose()

    output = Path("results/hnsw_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados guardados en {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=1000000)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--ef-values",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256],
    )

    args = parser.parse_args()

    if (
        args.size <= 0
        or args.queries <= 0
        or args.k <= 0
        or args.k > args.size
        or args.runs <= 0
        or args.warmup < 0
        or any(ef < args.k for ef in args.ef_values)
    ):
        parser.error("Parámetros inválidos")

    asyncio.run(main(args))
