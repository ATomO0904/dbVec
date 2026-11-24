from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = "postgresql+asyncpg://benchmark:benchmark@localhost:5433/vectors"

engine = create_async_engine(
    DATABASE_URL, pool_size=10, max_overflow=0, pool_pre_ping=False
)

session_local = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase): ...
