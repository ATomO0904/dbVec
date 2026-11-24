import asyncio
import numpy as np

from sqlalchemy import select
from qdrant_client import AsyncQdrantClient

from db.pgvector_db import session_local
from models.models import Document


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")

    ids = [1, 100, 1000, 5000, 10000]

    try:
        async with session_local() as session:
            result = await session.execute(
                select(Document.id, Document.embedding).where(Document.id.in_(ids))
            )

            pg_vectors = {row.id: np.asarray(row.embedding) for row in result.all()}

        points = await client.retrieve(
            collection_name="documents",
            ids=ids,
            with_vectors=True,
        )

        qdrant_vectors = {point.id: np.asarray(point.vector) for point in points}

        for id_ in ids:
            if id_ not in pg_vectors or id_ not in qdrant_vectors:
                print(f"ID {id_}: no existe en ambas bases")
                continue

            same = np.allclose(
                pg_vectors[id_],
                qdrant_vectors[id_],
                rtol=1e-5,
                atol=1e-6,
            )

            difference = np.max(np.abs(pg_vectors[id_] - qdrant_vectors[id_]))

            print(f"ID {id_}: iguales={same}, max_diff={difference:.10f}")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
