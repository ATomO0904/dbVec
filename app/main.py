from typing import Literal
from fastapi import Depends, FastAPI
from sentence_transformers import backend
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import session
from lifespan import lifespan
from db.pgvector_db import session_local
from repository.pgvector_repository import PgVectorRepository
from repository.qdrant_repository import QdrantRepository
from schemas.schemas import DocumentCreate, SearchRequest
from utils.embeddings import generate_embedding

app = FastAPI(
    title="Qdrant vs PgVector",
    lifespan=lifespan,
)


async def get_session():
    async with session_local() as session:
        yield session


def get_qdrant():
    return QdrantRepository(app.state.qdrant)


@app.post("/{backend}/documents")
async def insert_documents(
    backend: Literal["qdrant", "pgvector"],
    document: DocumentCreate,
    session: AsyncSession = Depends(get_session),
):
    embedding = await run_in_threadpool(generate_embedding, document.text)

    if backend == "qdrant":
        repository = get_qdrant()
    else:
        repository = PgVectorRepository(session)

    await repository.insert(
        document.id,
        document.text,
        embedding,
    )

    return {"status": "created", "backend": backend}


@app.post("/{backend}/search")
async def search_documents(
    backend: Literal["qdrant", "pgvector"],
    request: SearchRequest,
    session: AsyncSession = Depends(get_session),
):
    embedding = await run_in_threadpool(generate_embedding, request.query)

    if backend == "qdrant":
        repository = get_qdrant()
    else:
        repository = PgVectorRepository(session)

    return await repository.search(
        embedding,
        request.limit,
    )

