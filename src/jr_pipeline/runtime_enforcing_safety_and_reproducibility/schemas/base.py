"""Base class for all artifact schemas."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ArtifactSchema(BaseModel):
    """Base for durable artifact schemas; bump schema_version major on breaking changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: str = "1.0"
