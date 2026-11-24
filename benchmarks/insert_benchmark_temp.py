"""Insercion masiva con objetos temporales (no destruye el dataset de 1M).

Misma metodologia que `insert_benchmark.py`, pero usa las tablas/colecciones
`insert_bench_documents` y escribe un CSV propio. Al terminar elimina ambos
objetos temporales; la tabla `documents` y la coleccion `documents` quedan
intactas.

Uso:
    # Con indice durante la carga (metodologia original)
    python -m benchmarks.insert_benchmark_temp \
        --size 10000 --batch-size 100 --runs 3

    # Sin indice durante la carga y construccion despues
    python -m benchmarks.insert_benchmark_temp \
        --size 10000 --batch-size 100 --runs 3 --index-after
"""

import argparse
import asyncio
import csv
import gc
import statistics
import time
from pathlib import Path

from pgvector.sqlalchemy import VECTOR
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text

from benchmarks.dataset import generate_batches, generate_dataset
from db.pgvector_db import engine
from models.models import DIMENSIONS

PG_TABLE = "insert_bench_documents"
QDRANT_COLLECTION = "insert_bench_documents"
INDEX_NAME = "idx_insert_bench_hnsw"
DEFAULT_OUTPUT = Path("results/insert_results_10000.csv")
DEFAULT_INDEX_AFTER_OUTPUT = Path("results/insert_index_after_results.csv")


def log(message: str, started: float) -> None:
    print(f"[{time.perf_counter() - started:6.1f}s] {message}", flush=True)


def build_table(with_index: bool = True) -> Table:
    metadata = MetaData()
    table = Table(
        PG_TABLE,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("text", Text, nullable=False),
        Column("embedding", VECTOR(DIMENSIONS), nullable=False),
    )
    # Igual que models/models.py: el indice se mantiene durante la insercion.
    # Con --index-after se omite aqui y se crea al terminar la carga.
    if with_index:
        Index(
            INDEX_NAME,
            table.c.embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )
    return table


async def create_pg_index() -> float:
    """Crea el indice HNSW despues de la carga y devuelve los segundos."""
    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL maintenance_work_mem = '512MB'"))
        start = time.perf_counter()
        await conn.execute(text(
            f"CREATE INDEX {INDEX_NAME} ON {PG_TABLE} "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 100)"
        ))
        return time.perf_counter() - start


async def measure_pgvector(
    table: Table, vectors, batch_size: int, started: float,
    index_after: bool = False,
):
    async with engine.begin() as conn:
        if index_after:
            # El run anterior pudo dejar el indice creado; lo quitamos para
            # que la carga de este run vuelva a medirse sin el.
            await conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
        await conn.execute(text(f"TRUNCATE TABLE {PG_TABLE}"))
    gc.collect()

    latencies = []
    count = 0
    timer = time.perf_counter()
    for batch in generate_batches(vectors, batch_size):
        start = time.perf_counter()
        async with engine.begin() as conn:
            await conn.execute(table.insert(), batch)
        latencies.append(time.perf_counter() - start)
        count += len(batch)
        if count % 2000 == 0 or count == len(vectors):
            log(f"    pgvector {count}/{len(vectors)} insertados", started)
    total = time.perf_counter() - timer

    async with engine.begin() as conn:
        stored = await conn.scalar(text(f"SELECT COUNT(*) FROM {PG_TABLE}"))
    if stored != count:
        raise RuntimeError(f"pgvector: esperados {count}, encontrados {stored}")

    index_seconds = 0.0
    if index_after:
        log("    pgvector construyendo indice HNSW despues de la carga...", started)
        index_seconds = await create_pg_index()
        log(f"    pgvector indice creado en {index_seconds:.2f} s", started)

    return latencies, count, total, index_seconds


async def measure_qdrant(
    client, vectors, batch_size: int, started: float,
    index_after: bool = False,
):
    if await client.collection_exists(QDRANT_COLLECTION):
        await client.delete_collection(QDRANT_COLLECTION)
    await client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=DIMENSIONS, distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
        # Normalmente bajamos el umbral para que 10k se indexen como en 1M.
        # Con --index-after lo dejamos alto durante la carga y lo bajamos
        # despues, de modo que la indexacion no se cuele en el cronometro.
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=1_000_000 if index_after else 100,
            default_segment_number=1,
        ),
    )
    gc.collect()

    latencies = []
    count = 0
    timer = time.perf_counter()
    for batch in generate_batches(vectors, batch_size):
        points = [
            models.PointStruct(
                id=row["id"], vector=row["embedding"],
                payload={"text": row["text"]},
            )
            for row in batch
        ]
        start = time.perf_counter()
        await client.upsert(
            collection_name=QDRANT_COLLECTION, wait=True, points=points
        )
        latencies.append(time.perf_counter() - start)
        count += len(batch)
        if count % 2000 == 0 or count == len(vectors):
            log(f"    Qdrant {count}/{len(vectors)} insertados", started)
    total = time.perf_counter() - timer

    stored = (await client.count(
        collection_name=QDRANT_COLLECTION, exact=True
    )).count
    if stored != count:
        raise RuntimeError(f"Qdrant: esperados {count}, encontrados {stored}")

    index_seconds = 0.0
    if index_after:
        log("    Qdrant construyendo indice HNSW despues de la carga...", started)
        await client.update_collection(
            collection_name=QDRANT_COLLECTION,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100, default_segment_number=1
            ),
        )
        index_timer = time.perf_counter()
        while True:
            info = await client.get_collection(QDRANT_COLLECTION)
            if info.indexed_vectors_count >= count:
                break
            if time.perf_counter() - index_timer > 120:
                raise RuntimeError(
                    f"Qdrant no termino de indexar: "
                    f"{info.indexed_vectors_count}/{count}"
                )
            await asyncio.sleep(0.2)
        index_seconds = time.perf_counter() - index_timer
        log(
            f"    Qdrant indice creado en {index_seconds:.2f} s "
            f"({info.indexed_vectors_count}/{count} indexados)",
            started,
        )

    return latencies, count, total, index_seconds


async def main(args):
    started = time.perf_counter()
    mode = "indice DESPUES" if args.index_after else "indice ANTES"
    log(
        f"INICIO: size={args.size}, batch={args.batch_size}, "
        f"runs={args.runs} | {mode}",
        started,
    )

    vectors = generate_dataset(size=args.size, seed=42)
    log(f"Dataset generado: {vectors.shape}", started)

    table = build_table(with_index=not args.index_after)
    client = AsyncQdrantClient(url="http://localhost:6333", timeout=120)
    results = []

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(lambda c: table.create(c, checkfirst=False))
        log(
            f"Tabla temporal {PG_TABLE} creada "
            f"({'sin indice HNSW' if args.index_after else 'con indice HNSW'})",
            started,
        )

        for run in range(args.runs):
            order = (
                ["qdrant", "pgvector"] if run % 2 == 0
                else ["pgvector", "qdrant"]
            )
            for backend in order:
                log(f"Run {run + 1}/{args.runs} | {backend}", started)
                if backend == "pgvector":
                    latencies, count, total, index_seconds = await measure_pgvector(
                        table, vectors, args.batch_size, started, args.index_after
                    )
                else:
                    latencies, count, total, index_seconds = await measure_qdrant(
                        client, vectors, args.batch_size, started, args.index_after
                    )

                result = {
                    "backend": backend,
                    "vectors": count,
                    "batch_size": args.batch_size,
                    "total_seconds": round(total, 4),
                    "vectors_per_second": round(count / total, 2),
                    "mean_batch_ms": round(
                        statistics.mean(latencies) * 1000, 3
                    ),
                    "max_batch_ms": round(max(latencies) * 1000, 3),
                    "verified_count": count,
                    "run": run + 1,
                }
                if args.index_after:
                    total_with_index = total + index_seconds
                    result["index_build_seconds"] = round(index_seconds, 4)
                    result["total_with_index_seconds"] = round(total_with_index, 4)
                    result["vectors_per_second_with_index"] = round(
                        count / total_with_index, 2
                    )
                results.append(result)
                log(
                    f"    -> {count / total:,.2f} vec/s (solo carga) | "
                    f"media {result['mean_batch_ms']:.2f} ms | "
                    f"total {total:.2f} s"
                    + (
                        f" + {index_seconds:.2f} s de indice"
                        if args.index_after else ""
                    ),
                    started,
                )
    finally:
        log("Limpiando objetos temporales...", started)
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE}"))
        if await client.collection_exists(QDRANT_COLLECTION):
            await client.delete_collection(QDRANT_COLLECTION)
        await client.close()
        await engine.dispose()

    output = args.output or (
        DEFAULT_INDEX_AFTER_OUTPUT if args.index_after else DEFAULT_OUTPUT
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    log(f"FIN: resultados en {output}", started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--index-after", action="store_true",
        help="cargar sin indice HNSW y construirlo despues de medir la carga",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.size <= 0 or args.batch_size <= 0 or args.runs <= 0:
        parser.error("Los parametros deben ser positivos")

    asyncio.run(main(args))
