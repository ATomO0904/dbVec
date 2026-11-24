import asyncio
import numpy as np

from sqlalchemy import select, text
from qdrant_client import AsyncQdrantClient, models

from db.pgvector_db import session_local, engine
from models.models import Document


async def main():
    rng = np.random.default_rng(2026)

    query = rng.standard_normal(384, dtype=np.float32)
    query /= np.linalg.norm(query)

    vector = query.tolist()
    k = 10

    client = AsyncQdrantClient(url="http://localhost:6333")

    try:
        async with session_local() as session:
            distance = Document.embedding.cosine_distance(vector)

            statement = select(Document.id).order_by(distance).limit(k)

            # PostgreSQL exacto
            await session.execute(text("SET LOCAL enable_indexscan = off"))
            await session.execute(text("SET LOCAL enable_bitmapscan = off"))

            result = await session.execute(statement)
            pg_exact = result.scalars().all()

            await session.rollback()

            # PostgreSQL HNSW
            await session.execute(text("SET LOCAL hnsw.ef_search = 1000"))

            result = await session.execute(statement)
            pg_hnsw = result.scalars().all()

            await session.rollback()

        # Qdrant exacto
        result = await client.query_points(
            collection_name="documents",
            query=vector,
            limit=k,
            search_params=models.SearchParams(exact=True),
            with_payload=False,
        )

        qdrant_exact = [point.id for point in result.points]

        def agreement(a, b):
            return len(set(a) & set(b)) / k

        print("\nPostgreSQL exacto:", pg_exact)
        print("PostgreSQL HNSW:", pg_hnsw)
        print("Qdrant exacto:", qdrant_exact)

        print("\nPG exacto vs Qdrant exacto:", agreement(pg_exact, qdrant_exact))

        print("PG HNSW vs PG exacto:", agreement(pg_hnsw, pg_exact))

    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
