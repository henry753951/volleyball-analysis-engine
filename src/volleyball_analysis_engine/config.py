"""Environment configuration for the outbound worker process."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings loaded from VOLLYAI_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="VOLLYAI_",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    server_ws_url: str = "ws://localhost:4000/api/v1/ai/providers/ws"
    integration_id: str = Field(default="", min_length=1)
    token: str = Field(default="", min_length=16)
    fixture_root: Path = Path("fixtures")
    workspace: Path = Path("workspaces")
    provider_build_id: str = "volleyball-analysis-engine/0.1.0"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    tracking_variant: str = "sam-deep-eiou"

    def provider_url(self) -> str:
        """Return the central provider URL with its required integration identifier."""
        parsed = urlparse(self.server_ws_url)
        query = urlencode({"integration_id": self.integration_id})
        return urlunparse(parsed._replace(query=query))
