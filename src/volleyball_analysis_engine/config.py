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

    server_ws_url: str = "ws://localhost:4000/api/v2/ai/providers/ws"
    token: str = ""
    workspace: Path = Path("workspaces")
    provider_build_id: str = "volleyball-analysis-engine/0.9.0+multitask-v2"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    device: str = "cuda:0"
    detector_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    reid_every: int = Field(default=1, ge=1, le=30)
    multitask_sdk_root: Path = Path("src")
    multitask_checkpoint: Path = Path(".models/volleyball_multitask/best.pth")
    multitask_config: Path | None = None
    multitask_batch_size: int = Field(default=4, ge=1, le=64)
    multitask_fp16: bool = True
    multitask_warmup: bool = True
    local_tracker: Literal["deep_eiou", "harmonic"] = "deep_eiou"
    local_sam3_enabled: bool = True
    local_sam3_python: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation/.venv/Scripts/python.exe"
    )
    local_sam3_bridge: Path = Path("scripts/run_selective_sam3.py")
    local_sam3_timeout_seconds: int = Field(default=1800, ge=60, le=7200)
    # Provider Work v2 feature extraction is capability-gated for staged rollout.
    reid_feature_enabled: bool = False
    reid_association_enabled: bool = False
    identity_preview_enabled: bool = False
    reid_feature_batch_size: int = Field(default=32, ge=1, le=256)
    reid_feature_candidate_frames: int = Field(default=60, ge=6, le=240)
    reid_feature_selected_frames: int = Field(default=6, ge=1, le=24)
    reid_feature_min_frame_gap: int = Field(default=8, ge=1, le=120)
    multitask_pose_keypoint_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    multitask_pose_minimum_keypoints: int = Field(default=4, ge=1, le=17)
    prewarm_models: bool = True
    write_debug_artifacts: bool = False
    smp_root: Path = Path("../volley-ai/upstream/selective-mask-propagation")
    osnet_checkpoint: Path = Path(
        "../volley-ai/upstream/selective-mask-propagation/"
        "selective_mask_propagation/osnet/checkpoints/sports_model.pth.tar-60"
    )
    dinov2_root: Path = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    dinov2_checkpoint: Path = (
        Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
    )
    kpr_python: Path = Path("../volley-reid/third_party/kpr/.venv/Scripts/python.exe")
    kpr_root: Path = Path("../volley-reid/third_party/kpr")
    kpr_checkpoint: Path = Path(
        "../volley-reid/third_party/kpr/pretrained_models/"
        "kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar"
    )
    kpr_bridge: Path = Path("scripts/extract_kpr_pair_features.py")

    def validate_online(self) -> None:
        """Validate credentials only for the networked worker mode."""
        if len(self.token) < 16:
            raise ValueError("VOLLYAI_TOKEN must contain at least 16 characters")
        if not self.server_ws_url.startswith(("ws://", "wss://")):
            raise ValueError("VOLLYAI_SERVER_WS_URL must use ws:// or wss://")
