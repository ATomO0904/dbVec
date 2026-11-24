import argparse
import asyncio
import csv
import gc
import statistics
import time
from pathlib import Path

from sqlalchemy import text
from qdrant_client import AsyncQdrantClient, models

from db.pgvector_db import Base, engine, session_local
from models.models import DIMENSIONS, Document
from repository.pgvector_repository import PgVectorRepository
from repository.qdrant_repository import (
    QdrantRepository,
    COLLECTION,
)
from benchmarks.dataset import (
    generate_dataset,
    generate_batches,
)


async def prepare_pgvector():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("TRUNCATE TABLE documents"))


async def prepare_qdrant(client):
    if await client.collection_exists(COLLECTION):
        await client.delete_collection(COLLECTION)

    await client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(
            size=DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )


async def benchmark_backend(
    backend: str,
    vectors,
    batch_size: int,
    client,
):
    latencies = []
    count = 0

    if backend == "pgvector":
        await prepare_pgvector()
    else:
        await prepare_qdrant(client)

    gc.collect()

    # Medimos únicamente la carga de datos
    start_total = time.perf_counter()

    for batch in generate_batches(vectors, batch_size):
        start = time.perf_counter()

        if backend == "pgvector":
            async with session_local() as session:
                repository = PgVectorRepository(session)
                await repository.insert_batch(batch)
        else:
            repository = QdrantRepository(client)
            await repository.insert_batch(batch)

        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        count += len(batch)

    total_time = time.perf_counter() - start_total

    # Verificación de cantidad
    if backend == "pgvector":
        async with session_local() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM documents"))
            stored = result.scalar_one()
    else:
        result = await client.count(
            collection_name=COLLECTION,
            exact=True,
        )
        stored = result.count

    if stored != count:
        raise RuntimeError(
            f"{backend}: se esperaban {count}, pero se encontraron {stored}"
        )

    return {
        "backend": backend,
        "vectors": count,
        "batch_size": batch_size,
        "total_seconds": round(total_time, 4),
        "vectors_per_second": round(count / total_time, 2),
        "mean_batch_ms": round(statistics.mean(latencies) * 1000, 3),
        "max_batch_ms": round(max(latencies) * 1000, 3),
        "verified_count": stored,
    }


async def main(args):
    vectors = generate_dataset(
        size=args.size,
        seed=42,
    )

    client = AsyncQdrantClient(
        url="http://localhost:6333",
        timeout=120,
    )

    results = []

    try:
        # Alternamos el orden en ejecuciones sucesivas
        for run in range(args.runs):
            order = ["qdrant", "pgvector"] if run % 2 == 0 else ["pgvector", "qdrant"]

            for backend in order:
                print(f"\nRun {run + 1} | {backend} | N={args.size}")

                result = await benchmark_backend(
                    backend=backend,
                    vectors=vectors,
                    batch_size=args.batch_size,
                    client=client,
                )

                result["run"] = run + 1
                results.append(result)

                print(result)

    finally:
        await client.close()
        await engine.dispose()

    output = Path("results/insert_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)

    args = parser.parse_args()

    if args.size <= 0 or args.batch_size <= 0 or args.runs <= 0:
        parser.error("Los parámetros deben ser positivos")

    asyncio.run(main(args))
