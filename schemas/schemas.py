from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    id: int = Field(ge=1)
    text: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    id: int
    text: str
    score: float
