from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Index, Integer, Text
from pgvector.sqlalchemy import VECTOR

from db.pgvector_db import Base

DIMENSIONS = 384


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(DIMENSIONS))


hnsw_index = Index(
    "idx_documents_embedding_hnsw",
    Document.embedding,
    postgresql_using="hnsw",
    postgresql_with={
        "m": 16,
        "ef_construction": 100,
    },
    postgresql_ops={
        "embedding": "vector_cosine_ops",
    },
)
