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
    integration_id: str = ""
    token: str = ""
    workspace: Path = Path("workspaces")
    provider_build_id: str = "volleyball-analysis-engine/0.2.0"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    device: str = "cuda:0"
    rtv4_backend: str = "rolling"
    detector_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    court_stride: int = Field(default=30, ge=1, le=600)
    disable_amp: bool = False
    rtv4_root: Path = Path(".artifacts/rtv4")
    rtv4_config: Path = Path(
        ".artifacts/rtv4/configs/rtv4/rtv4_x3d_volleyball_v4a_decoupled.yml"
    )
    rtv4_checkpoint: Path = Path(".artifacts/models/best_stg1.pth")
    court_checkpoint: Path = Path(".artifacts/models/court-keypoints.pt")
    smp_root: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation"
    )
    osnet_checkpoint: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation/"
        "selective_mask_propagation/osnet/checkpoints/sports_model.pth.tar-60"
    )

    def provider_url(self) -> str:
        """Return the central provider URL with its required integration identifier."""
        if not self.integration_id:
            raise ValueError("VOLLYAI_INTEGRATION_ID is required in online worker mode")
        parsed = urlparse(self.server_ws_url)
        query = urlencode({"integration_id": self.integration_id})
        return urlunparse(parsed._replace(query=query))

    def validate_online(self) -> None:
        """Validate credentials only for the networked worker mode."""
        if len(self.token) < 16:
            raise ValueError("VOLLYAI_TOKEN must contain at least 16 characters")
        self.provider_url()
