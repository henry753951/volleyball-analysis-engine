"""Download public worker model assets and validate the bundled multitask SDK."""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

DINO_ARCHIVE_URL = "https://codeload.github.com/facebookresearch/dinov2/tar.gz/refs/heads/main"
DINO_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
)
SMP_ARCHIVE_URL = (
    "https://codeload.github.com/holma91/selective-mask-propagation/tar.gz/refs/heads/main"
)
OSNET_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/"
    "checkpoints/osnet_sports.pth.tar?download=true"
)
KPR_ARCHIVE_URL = "https://codeload.github.com/VlSomers/keypoint_promptable_reidentification/tar.gz/refs/heads/main"
KPR_CHECKPOINT_NAME = "kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar"


def validate_file(path: Path, label: str) -> None:
    """Require a non-empty model or source asset."""
    if not path.is_file() or path.stat().st_size == 0:
        message = f"{label} is missing: {path}"
        raise FileNotFoundError(message)


def download(url: str, destination: Path, label: str) -> None:
    """Atomically download an HTTPS asset before replacing the destination."""
    if destination.is_file():
        validate_file(destination, label)
        print(f"reuse {label}: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if urllib.parse.urlparse(url).scheme != "https":
        message = f"{label} download URL must use HTTPS: {url}"
        raise ValueError(message)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        print(f"download {label}: {url}")
        request = urllib.request.Request(
            url, headers={"User-Agent": "vollyai-worker-bootstrap/0.9"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=1024 * 1024)
        validate_file(temporary, label)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a tar archive while rejecting traversal outside the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                message = f"unsafe tar member: {member.name}"
                raise ValueError(message)
        bundle.extractall(destination)  # noqa: S202


def download_repository(archive_url: str, destination: Path, label: str) -> None:
    """Download and unpack a public source archive without requiring Git."""
    marker = destination / ".vollyai-source-complete"
    if marker.is_file():
        print(f"reuse {label}: {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        message = f"{label} destination is not an empty archive directory: {destination}"
        raise FileExistsError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    archive = temporary_root / "source.tar.gz"
    unpacked = temporary_root / "unpacked"
    try:
        download(archive_url, archive, f"{label} source archive")
        safe_extract_tar(archive, unpacked)
        roots = [path for path in unpacked.iterdir() if path.is_dir()]
        if len(roots) != 1:
            message = f"unexpected {label} archive layout"
            raise ValueError(message)
        roots[0].replace(destination)
        marker.touch()
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def prepare_multitask_sdk(
    *, project_root: Path, assets_root: Path, checkpoint_url: str | None
) -> tuple[Path, Path]:
    """Resolve the bundled SDK and download its checkpoint into the model cache."""
    sdk_root = project_root / "src"
    if not (sdk_root / "volleyball_sdk" / "__init__.py").is_file():
        message = f"bundled Volleyball SDK package is missing: {sdk_root / 'volleyball_sdk'}"
        raise FileNotFoundError(message)
    checkpoint = assets_root / "volleyball_multitask" / "best.pth"
    if checkpoint_url:
        download(checkpoint_url, checkpoint, "Volleyball multitask checkpoint")
    validate_file(checkpoint, "Volleyball multitask checkpoint")
    return sdk_root, checkpoint


def main() -> None:
    """Prepare the base models and optional nested-part ReID stack."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument(
        "--multitask-checkpoint-url", default=os.environ.get("VOLLYAI_MULTITASK_CHECKPOINT_URL")
    )
    parser.add_argument("--osnet-url")
    parser.add_argument("--dino-url")
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--kpr-checkpoint", type=Path)
    parser.add_argument("--kpr-checkpoint-url")
    args = parser.parse_args()

    assets_root = args.assets_root.resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    multitask_root, multitask_checkpoint = prepare_multitask_sdk(
        project_root=Path(__file__).resolve().parents[1],
        assets_root=assets_root,
        checkpoint_url=args.multitask_checkpoint_url,
    )

    smp_root = assets_root / "selective-mask-propagation"
    download_repository(SMP_ARCHIVE_URL, smp_root, "Selective Mask Propagation")
    osnet_checkpoint = (
        smp_root
        / "selective_mask_propagation"
        / "osnet"
        / "checkpoints"
        / "sports_model.pth.tar-60"
    )
    download(args.osnet_url or OSNET_CHECKPOINT_URL, osnet_checkpoint, "Sports OSNet")

    result = {
        "multitask_sdk_root": str(multitask_root),
        "multitask_checkpoint": str(multitask_checkpoint),
        "smp_root": str(smp_root),
        "osnet_checkpoint": str(osnet_checkpoint),
    }
    if args.with_reid:
        dino_root = assets_root / "dinov2"
        download_repository(DINO_ARCHIVE_URL, dino_root, "DINOv2")
        dino_checkpoint = assets_root / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth"
        download(args.dino_url or DINO_CHECKPOINT_URL, dino_checkpoint, "DINOv2 ViT-S/14")

        kpr_root = assets_root / "kpr"
        download_repository(KPR_ARCHIVE_URL, kpr_root, "KPR")
        kpr_destination = kpr_root / "pretrained_models" / KPR_CHECKPOINT_NAME
        if args.kpr_checkpoint:
            validate_file(
                args.kpr_checkpoint,
                "KPR checkpoint",
            )
            kpr_destination.parent.mkdir(parents=True, exist_ok=True)
            if not kpr_destination.exists():
                shutil.copy2(args.kpr_checkpoint, kpr_destination)
        elif args.kpr_checkpoint_url:
            download(
                args.kpr_checkpoint_url,
                kpr_destination,
                "KPR checkpoint",
            )
        else:
            validate_file(kpr_destination, "KPR checkpoint")
        result.update(
            {
                "dinov2_root": str(dino_root),
                "dinov2_checkpoint": str(dino_checkpoint),
                "kpr_root": str(kpr_root),
                "kpr_checkpoint": str(kpr_destination),
            }
        )

    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
