"""Environment configuration for the outbound worker process."""

from __future__ import annotations

from pathlib import Path

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
    token: str = ""
    workspace: Path = Path("workspaces")
    provider_build_id: str = "volleyball-analysis-engine/0.4.1+court-canonical-v4"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    device: str = "cuda:0"
    rtv4_backend: str = "rolling"
    detector_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    court_stride: int = Field(default=1, ge=1, le=600)
    court_imgsz: int = Field(default=1280, ge=320, le=2048)
    disable_amp: bool = False
    rtv4_root: Path = Path(".artifacts/rtv4")
    rtv4_config: Path = Path(
        ".artifacts/rtv4/configs/rtv4/rtv4_x3d_volleyball_v4a_decoupled.yml"
    )
    rtv4_checkpoint: Path = Path(".artifacts/models/best_stg1.pth")
    court_checkpoint: Path = Path(
        ".artifacts/models/court-keypoints-video91-canonical-v4.pt"
    )
    smp_root: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation"
    )
    osnet_checkpoint: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation/"
        "selective_mask_propagation/osnet/checkpoints/sports_model.pth.tar-60"
    )

    def validate_online(self) -> None:
        """Validate credentials only for the networked worker mode."""
        if len(self.token) < 16:
            raise ValueError("VOLLYAI_TOKEN must contain at least 16 characters")
        if not self.server_ws_url.startswith(("ws://", "wss://")):
            raise ValueError("VOLLYAI_SERVER_WS_URL must use ws:// or wss://")
