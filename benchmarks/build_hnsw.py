import asyncio
import time

from sqlalchemy import text

from db.pgvector_db import engine
from models.models import hnsw_index


async def main():
    start = time.perf_counter()

    async with engine.begin() as conn:
        await conn.execute(text("SET LOCAL maintenance_work_mem = '512MB'"))
        await conn.run_sync(
            lambda sync_conn: hnsw_index.create(
                sync_conn,
                checkfirst=True,
            )
        )

    elapsed = time.perf_counter() - start

    print(f"Índice HNSW construido en {elapsed:.2f} s")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
