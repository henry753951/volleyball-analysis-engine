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
    provider_build_id: str = "volleyball-analysis-engine/0.8.0+provider-work-v2-pose"
    instance_id: str | None = None
    max_concurrency: int = Field(default=1, ge=1, le=64)
    device: str = "cuda:0"
    rtv4_backend: str = "rolling"
    detector_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    detector_input_scale: float = Field(default=1.0, ge=0.5, le=1.0)
    reid_every: int = Field(default=1, ge=1, le=30)
    # Provider Work v2 feature extraction is capability-gated for staged rollout.
    reid_feature_enabled: bool = False
    reid_association_enabled: bool = False
    identity_preview_enabled: bool = False
    reid_feature_batch_size: int = Field(default=32, ge=1, le=256)
    reid_feature_candidate_frames: int = Field(default=60, ge=6, le=240)
    reid_feature_selected_frames: int = Field(default=6, ge=1, le=24)
    reid_feature_min_frame_gap: int = Field(default=8, ge=1, le=120)
    reid_vlm_enabled: bool = False
    reid_vlm_model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    reid_vlm_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    reid_vlm_max_new_tokens: int = Field(default=300, ge=64, le=2048)
    person_pose_enabled: bool = True
    person_pose_batch_size: int = Field(default=32, ge=1, le=256)
    person_pose_imgsz: int = Field(default=640, ge=320, le=2048)
    person_pose_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    person_pose_keypoint_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    person_pose_minimum_keypoints: int = Field(default=4, ge=1, le=17)
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
    dinov2_root: Path = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    dinov2_checkpoint: Path = (
        Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
    )
    pose_checkpoint: Path = Path("../volley-ai/yolo26n-pose.pt")
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
