import asyncio

from qdrant_client import AsyncQdrantClient, models

from repository.qdrant_repository import COLLECTION


async def main():
    client = AsyncQdrantClient(
        url="http://localhost:6333",
        timeout=120,
    )

    try:
        await client.update_collection(
            collection_name=COLLECTION,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )

        info = await client.get_collection(COLLECTION)

        print("Estado:", info.status)
        print("Optimizador:", info.optimizer_status)
        print("Vectores indexados:", info.indexed_vectors_count)
        print("Segmentos:", info.segments_count)

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
