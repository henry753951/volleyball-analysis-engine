"""Download and validate external worker source trees and checkpoints."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
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


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Extract a zip while rejecting traversal outside the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                message = f"unsafe zip member: {member.filename}"
                raise ValueError(message)
        bundle.extractall(destination)  # noqa: S202


def prepare_multitask_sdk(*, assets_root: Path, sdk_root: Path | None, sdk_url: str | None) -> Path:
    """Resolve the private unified SDK from an existing directory or downloadable zip."""
    if sdk_root is not None:
        resolved = sdk_root.resolve()
    elif sdk_url:
        archive = assets_root / "downloads" / "volleyball_inference_sdk.zip"
        download(sdk_url, archive, "Volleyball inference SDK archive")
        resolved = assets_root / "volleyball_inference_sdk"
        if not resolved.exists():
            safe_extract_zip(archive, resolved)
    else:
        resolved = assets_root / "volleyball_inference_sdk"
    if not (resolved / "volleyball_sdk" / "__init__.py").is_file():
        nested = [path.parent.parent for path in resolved.glob("*/volleyball_sdk/__init__.py")]
        if len(nested) == 1:
            resolved = nested[0]
    validate_file(resolved / "best.pth", "Volleyball multitask checkpoint")
    if not (resolved / "volleyball_sdk" / "__init__.py").is_file():
        message = f"Volleyball SDK package is missing: {resolved / 'volleyball_sdk'}"
        raise FileNotFoundError(message)
    return resolved


def main() -> None:
    """Prepare the base models and optional nested-part ReID stack."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--multitask-sdk-root", type=Path)
    parser.add_argument("--multitask-sdk-url")
    parser.add_argument("--osnet-url")
    parser.add_argument("--dino-url")
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--kpr-checkpoint", type=Path)
    parser.add_argument("--kpr-checkpoint-url")
    args = parser.parse_args()

    assets_root = args.assets_root.resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    multitask_root = prepare_multitask_sdk(
        assets_root=assets_root,
        sdk_root=args.multitask_sdk_root,
        sdk_url=args.multitask_sdk_url,
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
        "multitask_checkpoint": str(multitask_root / "best.pth"),
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
