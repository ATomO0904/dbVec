"""Benchmark HNSW con embeddings semanticos reales (AG News).

Descarga un dataset de texto moderado, lo embebe con all-MiniLM-L6-v2
(384 dimensiones, el modelo ya usado por la app) y repite la comparacion
pgvector vs Qdrant sobre datos con estructura semantica real.

Los objetos de base de datos son temporales (`semantic_bench`) y se eliminan
al terminar, por lo que no afectan al dataset de 1M de los otros benchmarks.
El resultado sobrescribe `results/hnsw_results.csv`, pero conserva la copia
histórica `results/hnsw_results_synthetic.csv` si ya existe.
"""

import argparse
import asyncio
import csv
import json
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import Column, Index, Integer, MetaData, Table, Text, text
from pgvector.sqlalchemy import VECTOR

from db.pgvector_db import engine
from models.models import DIMENSIONS

DATASET = "fancyzhx/ag_news"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PG_TABLE = "semantic_bench"
QDRANT_COLLECTION = "semantic_bench"
DATA_DIR = Path("data")
RESULTS = Path("results")


def fetch_rows(split: str, offset: int, length: int = 100) -> list[str]:
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET}&config=default&split={split}"
        f"&offset={offset}&length={length}"
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            return [row["row"]["text"] for row in payload["rows"]]
        except Exception as error:
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    return []


def get_texts(n_docs: int, n_queries: int) -> tuple[list[str], list[str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "ag_news_texts.json"
    documents: list[str] = []
    queries: list[str] = []
    if cache.exists():
        data = json.loads(cache.read_text())
        documents = data.get("documents", [])
        queries = data.get("queries", [])
        if len(documents) >= n_docs and len(queries) >= n_queries:
            print(
                f"Textos: usando caché ({len(documents)} docs, {len(queries)} consultas)",
                flush=True,
            )
            return documents[:n_docs], queries[:n_queries]

    print(f"Descargando {n_docs} documentos y {n_queries} consultas de AG News...", flush=True)
    for offset in range(len(documents), n_docs, 100):
        documents.extend(fetch_rows("train", offset, min(100, n_docs - offset)))
        if (offset // 100) % 10 == 0:
            cache.write_text(json.dumps({"documents": documents, "queries": queries}))
            print(f"  train: {len(documents)}/{n_docs}", flush=True)
        time.sleep(0.05)
    if not queries:
        queries = fetch_rows("test", 0, n_queries)
    cache.write_text(json.dumps({"documents": documents, "queries": queries}))
    return documents[:n_docs], queries[:n_queries]


def get_embeddings(documents, queries, n_docs, n_queries):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "ag_news_embeddings.npz"
    if cache.exists():
        with np.load(cache) as data:
            if data["doc"].shape[0] >= n_docs and data["query"].shape[0] >= n_queries:
                print("Embeddings: usando caché", flush=True)
                return data["doc"][:n_docs], data["query"][:n_queries]

    from sentence_transformers import SentenceTransformer

    print(f"Embebiendo con {MODEL}...", flush=True)
    model = SentenceTransformer(MODEL)
    doc = model.encode(
        documents, batch_size=128, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    query = model.encode(
        queries, batch_size=128, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    np.savez_compressed(cache, doc=doc, query=query)
    return doc, query


def build_metadata() -> Table:
    metadata = MetaData()
    table = Table(
        PG_TABLE,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("text", Text, nullable=False),
        Column("embedding", VECTOR(DIMENSIONS), nullable=False),
    )
    Index(
        "idx_semantic_bench_hnsw",
        table.c.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 100},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    return table


def percentiles(latencies: list[float]) -> dict:
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def log_step(message: str, started: float | None = None) -> None:
    prefix = f"[{time.perf_counter() - started:6.1f}s] " if started else ""
    print(prefix + message, flush=True)


async def exact_search(session, table: Table, vector: list[float], k: int) -> list[int]:
    # Fuerza secuencial para obtener el ground truth exacto.
    await session.execute(text("SET LOCAL enable_indexscan = off"))
    await session.execute(text("SET LOCAL enable_bitmapscan = off"))
    stmt = (
        table.select()
        .order_by(table.c.embedding.cosine_distance(vector))
        .limit(k)
    )
    result = await session.execute(stmt)
    ids = [row.id for row in result.all()]
    await session.rollback()
    return ids


async def pg_hnsw_search(session, table: Table, vector, limit: int, ef: int):
    # Con solo 10k filas el planificador prefiere seq scan; lo forzamos para
    # medir realmente el indice HNSW (igual que hace Qdrant con exact=False).
    await session.execute(text("SET LOCAL enable_seqscan = off"))
    await session.execute(
        text("SELECT set_config('hnsw.ef_search', :ef, true)"), {"ef": str(ef)}
    )
    stmt = (
        table.select()
        .order_by(table.c.embedding.cosine_distance(vector))
        .limit(limit)
    )
    result = await session.execute(stmt)
    ids = [row.id for row in result.all()]
    await session.rollback()
    return ids


async def qdrant_hnsw_search(client, vector, limit: int, ef: int):
    points = await client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=False,
        with_vectors=False,
        search_params=models.SearchParams(hnsw_ef=ef, exact=False),
    )
    return [point.id for point in points.points]


async def main(args):
    started = time.perf_counter()
    log_step(f"INICIO: size={args.size}, queries={args.queries}, runs={args.runs}", started)

    log_step("Paso 1/7: obteniendo textos (AG News)", started)
    documents, query_texts = get_texts(args.size, args.queries)
    log_step(f"  -> {len(documents)} documentos, {len(query_texts)} consultas", started)

    log_step("Paso 2/7: obteniendo embeddings (384 dims)", started)
    doc_vectors, query_vectors = get_embeddings(
        documents, query_texts, args.size, args.queries
    )
    log_step(f"  -> shapes {doc_vectors.shape}, {query_vectors.shape}", started)

    table = build_metadata()
    client = AsyncQdrantClient(url="http://localhost:6333", timeout=300)
    rows = []

    try:
        log_step("Paso 3/7: cargando vectores en pgvector", started)
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE}"))
            await conn.run_sync(lambda c: table.create(c, checkfirst=False))
            await conn.execute(table.insert(), [
                {"id": i + 1, "text": documents[i], "embedding": doc_vectors[i].tolist()}
                for i in range(args.size)
            ])
        log_step(f"  -> {args.size} filas insertadas en pgvector", started)

        log_step("Paso 4/7: cargando vectores en Qdrant", started)
        if await client.collection_exists(QDRANT_COLLECTION):
            await client.delete_collection(QDRANT_COLLECTION)
        await client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=DIMENSIONS, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16, ef_construct=100, full_scan_threshold=100
            ),
            # Con 1000 vectores el umbral por defecto (10000) hace que Qdrant
            # deje el segmento sin indexar. HNSW solo se construye cuando el
            # segmento supera indexing_threshold, asi que lo bajamos a 100.
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=100, default_segment_number=1
            ),
        )
        log_step(
            "  -> Qdrant HNSW: m=16, ef_construct=100, "
            "full_scan_threshold=100, indexing_threshold=100",
            started,
        )
        for start in range(0, args.size, 500):
            end = min(start + 500, args.size)
            await client.upsert(
                collection_name=QDRANT_COLLECTION,
                wait=True,
                points=[
                    models.PointStruct(
                        id=i + 1, vector=doc_vectors[i].tolist(),
                        payload={"text": documents[i]},
                    )
                    for i in range(start, end)
                ],
            )
            if end % 2000 == 0 or end == args.size:
                log_step(f"  -> Qdrant {end}/{args.size} upsertados", started)

        log_step("Paso 5/7: esperando a que Qdrant construya el índice HNSW", started)
        wait_started = time.perf_counter()
        while True:
            info = await client.get_collection(QDRANT_COLLECTION)
            indexed = info.indexed_vectors_count
            status = str(info.optimizer_status)
            if status == "ok" and indexed >= int(args.size * 0.9):
                break
            if time.perf_counter() - wait_started > 120:
                raise RuntimeError(
                    f"Qdrant no terminó de indexar: optimizer={status}, "
                    f"indexados={indexed}/{args.size}"
                )
            if int(time.perf_counter() - wait_started) % 5 == 0:
                log_step(
                    f"  -> optimizer={status}, indexados={indexed}/{args.size}",
                    started,
                )
            await asyncio.sleep(1)
        log_step(
            f"  -> listo: {info.points_count} puntos, {indexed} indexados",
            started,
        )

        log_step("Paso 6/7: verificando que pgvector usa el índice HNSW", started)
        async with engine.begin() as conn_check:
            await conn_check.execute(text("SET enable_seqscan = off"))
            plan = await conn_check.execute(
                text(
                    "EXPLAIN SELECT id FROM semantic_bench "
                    "ORDER BY embedding <=> '[" + ",".join(
                        f"{x:.6g}" for x in query_vectors[0].tolist()
                    ) + "]'::vector LIMIT 10"
                )
            )
            plan_text = " ".join(row[0] for row in plan)
            log_step(f"  -> plan: {plan_text[:160]}", started)
            if "hnsw" not in plan_text.lower():
                raise RuntimeError("pgvector no está usando el índice HNSW")

        # `engine.connect()` permite hacer rollback tras cada consulta para
        # que SET LOCAL no se filtre a la siguiente; con `engine.begin()` el
        # rollback cerraba la transacción del context manager.
        async with engine.connect() as conn:
            log_step("Paso 7/7: calculando ground truth exacto", started)
            ground_truth = []
            for i, vector in enumerate(query_vectors, start=1):
                ground_truth.append(
                    await exact_search(conn, table, vector.tolist(), args.k)
                )
                if i % 10 == 0 or i == len(query_vectors):
                    log_step(f"  -> ground truth {i}/{len(query_vectors)}", started)

            log_step("  -> midiendo recall y latencia", started)
            for run in range(args.runs):
                for ef in args.ef_values:
                    order = (
                        ["qdrant", "pgvector"]
                        if run % 2 == 0
                        else ["pgvector", "qdrant"]
                    )
                    run_results = {}
                    for backend in order:
                        log_step(
                            f"  -> run {run + 1}/{args.runs} | ef={ef} | "
                            f"midiendo {backend} ({len(query_vectors)} consultas)",
                            started,
                        )
                        latencies = []
                        recalls = []
                        warm = min(args.warmup, len(query_vectors))

                        for vector in query_vectors[:warm]:
                            if backend == "pgvector":
                                await pg_hnsw_search(
                                    conn, table, vector.tolist(), args.k, ef
                                )
                            else:
                                await qdrant_hnsw_search(
                                    client, vector.tolist(), args.k, ef
                                )

                        start_total = time.perf_counter()
                        for qi, (vector, expected) in enumerate(
                            zip(query_vectors, ground_truth), start=1
                        ):
                            start = time.perf_counter()
                            if backend == "pgvector":
                                actual = await pg_hnsw_search(
                                    conn, table, vector.tolist(), args.k, ef
                                )
                            else:
                                actual = await qdrant_hnsw_search(
                                    client, vector.tolist(), args.k, ef
                                )
                            latencies.append((time.perf_counter() - start) * 1000)
                            recalls.append(
                                len(set(actual) & set(expected)) / args.k
                            )
                            if qi % 25 == 0 or qi == len(query_vectors):
                                log_step(
                                    f"     {backend}: {qi}/{len(query_vectors)}",
                                    started,
                                )
                        total = time.perf_counter() - start_total

                        result = {
                            "backend": backend,
                            "ef_search": ef,
                            "recall_at_k": float(np.mean(recalls)),
                            "qps": len(query_vectors) / total,
                            **percentiles(latencies),
                            "run": run + 1,
                            "vectors": args.size,
                            "queries": args.queries,
                            "k": args.k,
                        }
                        run_results[backend] = result
                        rows.append(result)
                        print(
                            f"run={run + 1} {backend:8s} ef={ef:4d} "
                            f"recall={result['recall_at_k']:.4f} "
                            f"qps={result['qps']:.1f} "
                            f"p95={result['p95_ms']:.3f} ms",
                            flush=True,
                        )
    finally:
        log_step("Limpiando tabla/colección temporal...", started)
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE}"))
        if await client.collection_exists(QDRANT_COLLECTION):
            await client.delete_collection(QDRANT_COLLECTION)
        await client.close()
        await engine.dispose()

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "hnsw_results.csv"
    if output.exists():
        backup = RESULTS / "hnsw_results_synthetic.csv"
        if not backup.exists():
            shutil.copy2(output, backup)
            print(f"CSV sintético anterior guardado en {backup}")
        else:
            print(f"Backup histórico {backup} conservado (no se sobrescribe)")
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log_step(f"FIN: resultados en {output}", started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ef-values", nargs="+", type=int, default=[32, 64, 128, 256])
    args = parser.parse_args()

    if (
        args.size <= 0 or args.queries <= 0 or args.k <= 0 or args.k > args.size
        or args.runs <= 0 or args.warmup < 0
        or any(ef < args.k for ef in args.ef_values)
    ):
        parser.error("Parámetros inválidos")

    asyncio.run(main(args))
