from annotated_doc.main import Doc
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import session
from sqlalchemy import text as sql_text
from models.models import Document


class PgVectorRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session: AsyncSession = session

    async def insert(self, id: int, text: str, embedding: list[float]):
        document = await self.session.get(Document, id)

        if document is None:
            document = Document(id=id)
            self.session.add(document)

        document.text = text
        document.embedding = embedding

        await self.session.commit()

    async def search(self, embedding: list[float], limit: int = 5):
        distance = Document.embedding.cosine_distance(embedding)

        statement = (
            select(
                Document.id,
                Document.text,
                (1 - distance).label("score"),
            )
            .order_by(distance)
            .limit(limit=limit)
        )

        result = await self.session.execute(statement)

        return [
            {
                "id": row.id,
                "text": row.text,
                "score": float(row.score),
            }
            for row in result.all()
        ]

    async def insert_batch(self, documents: list[dict]):
        # no usamos session.add_all() para no sobrecargar el seguimiento de objetos
        statement = pg_insert(Document).values(documents)

        statement = statement.on_conflict_do_update(
            index_elements=[Document.id],
            set_={
                "text": statement.excluded.text,
                "embedding": statement.excluded.embedding,
            },
        )

        await self.session.execute(statement)
        await self.session.commit()

    async def search_hnsw(
        self, embedding: list[float], limit: int = 10, ef_search: int = 64
    ):
        await self.session.execute(
            sql_text("SELECT set_config('hnsw.ef_search', :ef, true)"),
            {"ef": str(ef_search)},
        )

        distance = Document.embedding.cosine_distance(embedding)

        statement = (
            select(Document.id, (1 - distance).label("score"))
            .order_by(distance)
            .limit(limit)
        )

        result = await self.session.execute(statement)

        return [{"id": row.id, "score": row.score} for row in result.all()]
