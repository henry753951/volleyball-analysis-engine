"""Download and validate external worker source trees and checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DINO_REPOSITORY = "https://github.com/facebookresearch/dinov2.git"
DINO_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth"
)
DINO_CHECKPOINT_SHA256 = "f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb"
SMP_REPOSITORY = "https://github.com/holma91/selective-mask-propagation.git"
OSNET_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/holma91/SAM-Deep-EIoU/resolve/main/"
    "checkpoints/osnet_sports.pth.tar?download=true"
)
OSNET_CHECKPOINT_SHA256 = "8d5b2fd8763db34c2aad69810466adf413f0426d9f8119d322227e0e639c5fbd"
KPR_REPOSITORY = "https://github.com/VlSomers/keypoint_promptable_reidentification.git"
KPR_CHECKPOINT_NAME = "kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar"
KPR_CHECKPOINT_SHA256 = "9bea1e6dd887fb7af8c2f154912cce846c8c809c4c2357df4b89282889b31a20"
MULTITASK_CHECKPOINT_SHA256 = "60ecd86921e13600b7de3f375bdc01ed4cbcd64330e1e2509b77e70f7bcc4ea3"


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, expected_sha256: str | None, label: str) -> None:
    """Require a non-empty file and optionally verify its pinned digest."""
    if not path.is_file() or path.stat().st_size == 0:
        message = f"{label} is missing: {path}"
        raise FileNotFoundError(message)
    if expected_sha256:
        actual = sha256(path)
        if actual.lower() != expected_sha256.lower():
            message = f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}: {path}"
            raise ValueError(message)


def download(url: str, destination: Path, expected_sha256: str | None, label: str) -> None:
    """Atomically download an asset and validate it before replacing the destination."""
    if destination.is_file():
        validate_file(destination, expected_sha256, label)
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
        validate_file(temporary, expected_sha256, label)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def clone(repository: str, destination: Path, label: str) -> None:
    """Clone a source tree once and validate its Git metadata on reuse."""
    if (destination / ".git").is_dir():
        print(f"reuse {label}: {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        message = f"{label} destination is not an empty Git checkout: {destination}"
        raise FileExistsError(message)
    destination.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        message = "Git executable was not found"
        raise FileNotFoundError(message)
    subprocess.run(  # noqa: S603
        [git, "clone", "--filter=blob:none", "--depth", "1", repository, str(destination)],
        check=True,
    )


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
        bundle.extractall(destination)


def prepare_multitask_sdk(
    *, assets_root: Path, sdk_root: Path | None, sdk_url: str | None, sdk_sha256: str
) -> Path:
    """Resolve the private unified SDK from an existing directory or downloadable zip."""
    if sdk_root is not None:
        resolved = sdk_root.resolve()
    elif sdk_url:
        archive = assets_root / "downloads" / "volleyball_inference_sdk.zip"
        download(sdk_url, archive, None, "Volleyball inference SDK archive")
        resolved = assets_root / "volleyball_inference_sdk"
        if not resolved.exists():
            safe_extract_zip(archive, resolved)
    else:
        resolved = assets_root / "volleyball_inference_sdk"
    if not (resolved / "volleyball_sdk" / "__init__.py").is_file():
        nested = [path.parent.parent for path in resolved.glob("*/volleyball_sdk/__init__.py")]
        if len(nested) == 1:
            resolved = nested[0]
    validate_file(resolved / "best.pth", sdk_sha256, "Volleyball multitask checkpoint")
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
    parser.add_argument("--multitask-sha256", default=MULTITASK_CHECKPOINT_SHA256)
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
        sdk_sha256=args.multitask_sha256,
    )

    smp_root = assets_root / "selective-mask-propagation"
    clone(SMP_REPOSITORY, smp_root, "Selective Mask Propagation")
    osnet_checkpoint = (
        smp_root
        / "selective_mask_propagation"
        / "osnet"
        / "checkpoints"
        / "sports_model.pth.tar-60"
    )
    download(OSNET_CHECKPOINT_URL, osnet_checkpoint, OSNET_CHECKPOINT_SHA256, "Sports OSNet")

    result = {
        "multitask_sdk_root": str(multitask_root),
        "multitask_checkpoint": str(multitask_root / "best.pth"),
        "smp_root": str(smp_root),
        "osnet_checkpoint": str(osnet_checkpoint),
    }
    if args.with_reid:
        dino_root = assets_root / "dinov2"
        clone(DINO_REPOSITORY, dino_root, "DINOv2")
        dino_checkpoint = assets_root / "checkpoints" / "dinov2_vits14_reg4_pretrain.pth"
        download(DINO_CHECKPOINT_URL, dino_checkpoint, DINO_CHECKPOINT_SHA256, "DINOv2 ViT-S/14")

        kpr_root = assets_root / "kpr"
        clone(KPR_REPOSITORY, kpr_root, "KPR")
        kpr_destination = kpr_root / "pretrained_models" / KPR_CHECKPOINT_NAME
        if args.kpr_checkpoint:
            validate_file(args.kpr_checkpoint, KPR_CHECKPOINT_SHA256, "KPR checkpoint")
            kpr_destination.parent.mkdir(parents=True, exist_ok=True)
            if not kpr_destination.exists():
                shutil.copy2(args.kpr_checkpoint, kpr_destination)
        elif args.kpr_checkpoint_url:
            download(
                args.kpr_checkpoint_url,
                kpr_destination,
                KPR_CHECKPOINT_SHA256,
                "KPR checkpoint",
            )
        else:
            validate_file(kpr_destination, KPR_CHECKPOINT_SHA256, "KPR checkpoint")
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
