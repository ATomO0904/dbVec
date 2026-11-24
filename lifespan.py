from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient, models
from sqlalchemy import text

from db.pgvector_db import Base, engine
from models.models import DIMENSIONS
from repository.qdrant_repository import COLLECTION


@asynccontextmanager
async def lifespan(app):
    # POSTGRESQL CON PGVECTOR
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    client = AsyncQdrantClient(url="http://localhost:6333")

    if not await client.collection_exists(COLLECTION):
        await client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )

    app.state.qdrant = client

    yield

    await client.close()
    await engine.dispose()
