from qdrant_client import AsyncQdrantClient, models
from sentence_transformers.cross_encoder import model
from sqlalchemy.engine import result

COLLECTION = "documents"


class QdrantRepository:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    async def insert(self, id: int, text: str, embedding: list[float]):
        await self.client.upsert(
            collection_name=COLLECTION,
            wait=True,
            points=[
                models.PointStruct(id=id, vector=embedding, payload={"text": text})
            ],
        )

    async def search(self, embedding: list[float], limit: int = 5):
        result = await self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(exact=True),
        )

        return [
            {"id": point.id, "text": point.payload["text"], "score": float(point.score)}
            for point in result.points
        ]

    async def insert_batch(self, documents: list[dict]):
        points = [
            models.PointStruct(
                id=doc["id"], vector=doc["embedding"], payload={"text": doc["text"]}
            )
            for doc in documents
        ]
        await self.client.upsert(
            collection_name=COLLECTION,
            wait=True,
            points=points,
        )

    async def search_hnsw(
        self, embedding: list[float], limit: int = 10, ef_search: int = 64
    ):
        result = await self.client.query_points(
            collection_name=COLLECTION,
            query=embedding,
            limit=limit,
            with_payload=False,
            with_vectors=False,
            search_params=models.SearchParams(hnsw_ef=ef_search, exact=False),
        )

        return [
            {"id": point.id, "score": float(point.score)} for point in result.points
        ]
