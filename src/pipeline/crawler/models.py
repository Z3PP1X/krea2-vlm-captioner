"""Data structures for crawler candidate items before ingestion."""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class CandidateImage(BaseModel):
    """Represents an image found on a web page before downloading."""

    source: str
    source_url: str
    page_url: str
    context_title: str = "Unknown Set"
    context_tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    trigger_word: Optional[str] = None
    estimated_width: Optional[int] = None
    estimated_height: Optional[int] = None
    http_headers: Dict[str, str] = Field(default_factory=dict)
