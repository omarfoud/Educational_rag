from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class CreateScan(BaseModel):
    source_language_code: str
    translation_language_code: str


class VocabularyWord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    id: str = Field(min_length=1, max_length=100)
    order: int = Field(ge=0)
    source_text: str = Field(max_length=500)
    translation_text: str = Field(max_length=500)
    status: Literal["ok", "needs_review"] = "ok"
    page_id: str | None = None


class ConfirmVocabulary(BaseModel):
    words: list[VocabularyWord] = Field(min_length=1, max_length=500)
