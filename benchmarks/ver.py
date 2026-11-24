import asyncio
from qdrant_client import AsyncQdrantClient


async def main():
    client = AsyncQdrantClient(
        url="http://localhost:6333",
        timeout=120,
    )

    info = await client.get_collection("documents")

    print("Points:", info.points_count)
    print("Indexed vectors:", info.indexed_vectors_count)
    print("Segments:", info.segments_count)
    print("Optimizer:", info.optimizer_status)

    info = await client.get_collection("documents")

    print("Estado:", info.status)
    print("Puntos:", info.points_count)
    print("Vectores indexados:", info.indexed_vectors_count)
    print("Segmentos:", info.segments_count)
    print("HNSW:", info.config.hnsw_config)


asyncio.run(main())
