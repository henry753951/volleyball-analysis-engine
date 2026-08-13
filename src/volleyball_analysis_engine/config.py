"""Environment configuration for the outbound worker process."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    provider_build_id: str = "volleyball-analysis-engine/0.6.0+reid-bank-court-v3-majority"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    device: str = "cuda:0"
    rtv4_backend: str = "rolling"
    detector_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    detector_input_scale: float = Field(default=1.0, ge=0.5, le=1.0)
    reid_every: int = Field(default=1, ge=1, le=30)
    court_model: str = "v3"
    court_imgsz: int = Field(default=512, ge=320, le=2048)
    court_batch_size: int = Field(default=16, ge=1, le=64)
    court_layout_every: int = Field(default=1, ge=1, le=600)
    court_refresh_every: int = Field(default=120, ge=1, le=600)
    court_track_every: int = Field(default=1, ge=1, le=30)
    court_max_hold_frames: int = Field(default=180, ge=0, le=600)
    court_decoder: Literal["auto", "spatial", "cuda"] = "auto"
    prewarm_models: bool = True
    write_debug_artifacts: bool = False
    disable_amp: bool = False
    rtv4_root: Path = Path(".artifacts/rtv4")
    rtv4_config: Path = Path(".artifacts/rtv4/configs/rtv4/rtv4_x3d_volleyball_v4a_decoupled.yml")
    rtv4_checkpoint: Path = Path(".artifacts/models/best_stg1.pth")
    smp_root: Path = Path("../volley-ai/upstream/selective-mask-propagation")
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
