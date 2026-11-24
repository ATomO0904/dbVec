"""Benchmark de consumo de RAM de pgvector y Qdrant.

Mide la memoria de los contenedores leyendo sus cgroups (v2), no solo el
page cache que muestra `docker stats`. Crea objetos temporales
(`mem_bench_documents`) para no afectar al dataset de los otros benchmarks.
"""

import argparse
import asyncio
import csv
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import Column, Integer, MetaData, Table, text
from pgvector.sqlalchemy import VECTOR

from benchmarks.dataset import generate_dataset
from db.pgvector_db import engine
from models.models import DIMENSIONS

PG_TABLE = "mem_bench_documents"
QDRANT_COLLECTION = "mem_bench_documents"


def cgroup_path(container: str) -> Path:
    cid = subprocess.check_output(
        ["docker", "inspect", "-f", "{{.Id}}", container], text=True
    ).strip()
    path = Path(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope")
    if not (path / "memory.current").exists():
        raise RuntimeError(f"No se encontró el cgroup v2 de {container}: {path}")
    return path


class MemorySampler:
    """Muestrea memory.current y memory.stat en background."""

    def __init__(self, paths: dict[str, Path], interval: float):
        self.paths = paths
        self.interval = interval
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            t = time.perf_counter()
            for backend, path in self.paths.items():
                try:
                    current = int((path / "memory.current").read_text())
                    stat = {}
                    for line in (path / "memory.stat").read_text().splitlines():
                        key, value = line.split()
                        stat[key] = int(value)
                    self.samples.append(
                        {
                            "t": t,
                            "backend": backend,
                            "cgroup_bytes": current,
                            "anon_bytes": stat.get("anon", 0),
                            "shmem_bytes": stat.get("shmem", 0),
                            "slab_bytes": stat.get("slab", 0),
                            "kernel_bytes": stat.get("kernel", 0),
                        }
                    )
                except (OSError, ValueError):
                    continue
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def values(
        self, backend: str, key: str, t_min: float = 0.0, t_max: float = float("inf")
    ) -> list[float]:
        return [
            s[key]
            for s in self.samples
            if s["backend"] == backend and t_min <= s["t"] <= t_max
        ]

    def rss(
        self, backend: str, t_min: float = 0.0, t_max: float = float("inf")
    ) -> list[float]:
        return [
            s["anon_bytes"] + s["shmem_bytes"] + s["slab_bytes"] + s["kernel_bytes"]
            for s in self.samples
            if s["backend"] == backend and t_min <= s["t"] <= t_max
        ]


def summarize(sampler: MemorySampler, backend: str, t_start: float, size: int) -> dict:
    def last(values, n=5):
        return statistics.median(values[-n:])

    rss_work = sampler.rss(backend, t_start)
    cgroup_work = sampler.values(backend, "cgroup_bytes", t_start)
    anon_work = sampler.values(backend, "anon_bytes", t_start)
    shmem_work = sampler.values(backend, "shmem_bytes", t_start)

    def pagecache(t_min=0.0, t_max=float("inf")):
        return [
            s["cgroup_bytes"]
            - s["anon_bytes"]
            - s["shmem_bytes"]
            - s["slab_bytes"]
            - s["kernel_bytes"]
            for s in sampler.samples
            if s["backend"] == backend and t_min <= s["t"] <= t_max
        ]

    # Baseline: muestras tomadas antes de empezar a cargar.
    base_rss = last(sampler.rss(backend, 0, t_start))
    base_cgroup = last(sampler.values(backend, "cgroup_bytes", 0, t_start))
    base_anon = last(sampler.values(backend, "anon_bytes", 0, t_start))
    base_shmem = last(sampler.values(backend, "shmem_bytes", 0, t_start))
    base_pagecache = last(pagecache(0, t_start))

    end_rss = last(sampler.rss(backend, t_start))
    end_cgroup = last(sampler.values(backend, "cgroup_bytes", t_start))
    end_anon = last(sampler.values(backend, "anon_bytes", t_start))

    return {
        "backend": backend,
        "vectors": size,
        "base_rss_mb": round(base_rss / 2**20, 2),
        "peak_rss_mb": round(max(rss_work) / 2**20, 2),
        "end_rss_mb": round(end_rss / 2**20, 2),
        "delta_peak_rss_mb": round((max(rss_work) - base_rss) / 2**20, 2),
        "delta_end_rss_mb": round((end_rss - base_rss) / 2**20, 2),
        "bytes_per_vector": round((end_rss - base_rss) / size, 1),
        "base_anon_mb": round(base_anon / 2**20, 2),
        "peak_anon_mb": round(max(anon_work) / 2**20, 2),
        "delta_peak_anon_mb": round((max(anon_work) - base_anon) / 2**20, 2),
        "base_shmem_mb": round(base_shmem / 2**20, 2),
        "peak_shmem_mb": round(max(shmem_work) / 2**20, 2),
        "delta_peak_shmem_mb": round((max(shmem_work) - base_shmem) / 2**20, 2),
        "base_pagecache_mb": round(base_pagecache / 2**20, 2),
        "peak_pagecache_mb": round(max(pagecache(t_start)) / 2**20, 2),
        "delta_peak_pagecache_mb": round(
            (max(pagecache(t_start)) - base_pagecache) / 2**20, 2
        ),
        "base_cgroup_mb": round(base_cgroup / 2**20, 2),
        "peak_cgroup_mb": round(max(cgroup_work) / 2**20, 2),
        "end_cgroup_mb": round(end_cgroup / 2**20, 2),
        "delta_peak_cgroup_mb": round((max(cgroup_work) - base_cgroup) / 2**20, 2),
        "delta_end_cgroup_mb": round((end_cgroup - base_cgroup) / 2**20, 2),
        "bytes_per_vector_cgroup": round((end_cgroup - base_cgroup) / size, 1),
    }


async def prepare_postgres():
    metadata = MetaData()
    table = Table(
        PG_TABLE,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("embedding", VECTOR(DIMENSIONS), nullable=False),
    )
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE}"))
        await conn.run_sync(lambda sync_conn: table.create(sync_conn, checkfirst=False))
    return table


async def prepare_qdrant(client):
    if await client.collection_exists(QDRANT_COLLECTION):
        await client.delete_collection(QDRANT_COLLECTION)
    await client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=DIMENSIONS, distance=models.Distance.COSINE
        ),
    )


async def load_postgres(table, vectors, batch_size):
    async with engine.begin() as conn:
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            rows = [
                {"id": i + 1, "embedding": vectors[i].tolist()}
                for i in range(start, end)
            ]
            await conn.execute(table.insert(), rows)

    # Construcción del índice en bloque, igual que hace Qdrant con sus segmentos.
    async with engine.begin() as conn:
        await conn.execute(text("SET maintenance_work_mem = '512MB'"))
        await conn.execute(
            text(
                f"CREATE INDEX idx_mem_bench_hnsw ON {PG_TABLE} "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 100)"
            )
        )


async def load_qdrant(client, vectors, batch_size):
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        await client.upsert(
            collection_name=QDRANT_COLLECTION,
            wait=True,
            points=[
                models.PointStruct(id=i + 1, vector=vectors[i].tolist())
                for i in range(start, end)
            ],
        )
    # Qdrant deja los segmentos pequeños sin indexar (full_scan_threshold),
    # así que indexed_vectors_count no llega exactamente al total.
    while True:
        info = await client.get_collection(QDRANT_COLLECTION)
        if (
            str(info.optimizer_status) == "ok"
            and info.indexed_vectors_count >= int(len(vectors) * 0.9)
        ):
            break
        await asyncio.sleep(1)
    await asyncio.sleep(2)


async def cleanup(client):
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP TABLE IF EXISTS {PG_TABLE}"))
    if await client.collection_exists(QDRANT_COLLECTION):
        await client.delete_collection(QDRANT_COLLECTION)


def plot(sampler: MemorySampler, t_start: float, rows: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    names = [r["backend"] for r in rows]
    vals = [r["delta_peak_rss_mb"] for r in rows]
    ax[0].bar(names, vals, color=["#2b8cbe", "#e6550d"])
    ax[0].set_title("RAM residente añadida (pico) al cargar 100k vectores")
    ax[0].set_ylabel("MiB sobre el estado vacío")
    for i, v in enumerate(vals):
        ax[0].text(i, v + max(vals) * 0.02, f"{v:.0f} MiB", ha="center", fontweight="bold")
    ax[0].grid(axis="y", alpha=0.3)

    for backend, color in (("pgvector", "#e6550d"), ("qdrant", "#2b8cbe")):
        times = [s["t"] - t_start for s in sampler.samples if s["backend"] == backend]
        mem = [
            (s["anon_bytes"] + s["shmem_bytes"] + s["slab_bytes"] + s["kernel_bytes"])
            / 2**20
            for s in sampler.samples
            if s["backend"] == backend
        ]
        ax[1].plot(times, mem, label=backend, color=color, lw=1.5)
    ax[1].set_title("Evolución de RAM residente durante el experimento")
    ax[1].set_xlabel("Segundos desde el inicio de la carga")
    ax[1].set_ylabel("MiB")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    fig.suptitle("Consumo de RAM: pgvector vs Qdrant", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = Path("results/memory_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Gráfica: {out}")


async def main(args):
    vectors = generate_dataset(args.size, seed=42)
    client = AsyncQdrantClient(url="http://localhost:6333", timeout=300)

    sampler = MemorySampler(
        {
            "pgvector": cgroup_path(args.postgres_container),
            "qdrant": cgroup_path(args.qdrant_container),
        },
        args.sample_interval,
    )
    sampler.start()

    try:
        table = await prepare_postgres()
        await prepare_qdrant(client)

        await asyncio.sleep(2)
        t_start = time.perf_counter()
        print(f"Cargando {args.size:,} vectores en pgvector...", flush=True)
        t0 = time.perf_counter()
        await load_postgres(table, vectors, args.batch_size)
        print(f"  pgvector: {time.perf_counter() - t0:.1f} s", flush=True)

        print(f"Cargando {args.size:,} vectores en Qdrant...", flush=True)
        t0 = time.perf_counter()
        await load_qdrant(client, vectors, args.batch_size)
        print(f"  qdrant: {time.perf_counter() - t0:.1f} s", flush=True)

        await asyncio.sleep(3)

        rows = [
            summarize(sampler, backend, t_start, args.size)
            for backend in ("pgvector", "qdrant")
        ]
        for row in rows:
            print(row, flush=True)

        out = Path("results/memory_results.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Resultados: {out}")

        samples = Path("results/memory_samples.csv")
        with samples.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=sampler.samples[0].keys())
            writer.writeheader()
            writer.writerows(sampler.samples)
        print(f"Muestras: {samples}")

        plot(sampler, t_start, rows)
    finally:
        await cleanup(client)
        await client.close()
        await engine.dispose()
        sampler.stop()
        print("Objetos temporales eliminados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--postgres-container", default="dbvec-postgres-1")
    parser.add_argument("--qdrant-container", default="dbvec-qdrant-1")
    args = parser.parse_args()

    if args.size <= 0 or args.batch_size <= 0 or args.sample_interval <= 0:
        parser.error("Los parámetros deben ser positivos")

    asyncio.run(main(args))
