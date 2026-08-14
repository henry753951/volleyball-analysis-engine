"""Tile the selected frames into one image for the VLM.

Upscaling is not cosmetic.  A player is roughly 150 px tall in a 720p broadcast frame, so
the jersey number occupies about 20-25 px.  A vision transformer patch is 14-16 px, so at
native scale the whole number falls inside one or two patches and carries no usable signal.
Resampling to 4x puts it across several patches.  Lanczos adds no information, it moves the
information that is already there onto a scale the model can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, cast

from PIL import Image

from .records import SelectedFrame

BACKGROUND = (18, 18, 18)
GUTTER = 6


@dataclass(frozen=True, slots=True)
class SheetSettings:
    """How the selected crops are tiled and how far they are upscaled."""

    scale: int = 4
    columns: int = 3
    include_torso_strip: bool = True
    torso_min_height: int = 180


def _load(path: str, scale: float, min_height: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if min_height and height * scale < min_height:
        scale = min_height / float(height)
    resample = cast("Any", Image.Resampling.LANCZOS)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), resample)


def build_contact_sheet(
    frames: list[SelectedFrame], settings: SheetSettings
) -> Image.Image | None:
    """Grid of full-body crops, with a zoomed torso strip underneath."""
    tiles = [_load(frame.full_path, settings.scale) for frame in frames if frame.full_path]
    if not tiles:
        return None
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    rows = ceil(len(tiles) / settings.columns)
    grid_width = settings.columns * tile_width + (settings.columns + 1) * GUTTER
    grid_height = rows * tile_height + (rows + 1) * GUTTER

    strip: list[Image.Image] = []
    if settings.include_torso_strip:
        strip = [
            _load(frame.torso_path, settings.scale, settings.torso_min_height)
            for frame in frames
            if frame.torso_path
        ]
    strip_height = (max(tile.height for tile in strip) + 2 * GUTTER) if strip else 0
    strip_width = (
        sum(tile.width for tile in strip) + (len(strip) + 1) * GUTTER if strip else 0
    )

    sheet = Image.new(
        "RGB", (max(grid_width, strip_width), grid_height + strip_height), BACKGROUND
    )
    for position, tile in enumerate(tiles):
        row, column = divmod(position, settings.columns)
        x = GUTTER + column * (tile_width + GUTTER) + (tile_width - tile.width) // 2
        y = GUTTER + row * (tile_height + GUTTER) + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
    x = GUTTER
    for tile in strip:
        sheet.paste(tile, (x, grid_height + GUTTER))
        x += tile.width + GUTTER
    return sheet
