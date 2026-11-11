from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FittingIdentifier:
    """A single identified fitting identifier within a image.

    Attributes:
        fitting_id: The identifier for the fitting, for example ``A`` or ``B-12``.
        x: The horizontal pixel coordinate of the fitting in the source image.
        y: The vertical pixel coordinate of the fitting in the source image.
        radius_px: The display radius used for interaction overlays.
    """

    fitting_id: str
    x: float
    y: float
    radius_px: float = 0.0

    def as_leaflet_point(self) -> dict[str, str | float]:
        """Return a Leaflet-friendly payload for this fitting identifier."""
        return {
            "id": self.fitting_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "radius_px": round(self.radius_px, 2),
        }



@dataclass(slots=True)
class ProcessingResult:
    """The output of locating fittings and writing associated raster assets.

    Attributes:
        image_path: The original image image path.
        output_dir: The directory storing generated outputs.
        width_px: Source image width in pixels.
        height_px: Source image height in pixels.
        positions: Fitting identifiers mapped to image coordinates.
        zoom_paths: Paths to the generated raster zoom layers.
        hitboxes_path: Optional path to the generated hitbox JSON file.
        svg_path: Optional path to the generated SVG copy of the image.
    """

    image_path: Path
    output_dir: Path
    width_px: int
    height_px: int
    positions: list[FittingIdentifier]
    zoom_paths: list[Path]
    hitboxes_path: Path | None = None
    svg_path: Path | None = None

    def to_leaflet_payload(self) -> dict[str, Any]:
        """Return the JSON-serializable payload expected by the frontend map."""
        return {
            "image": {
                "source": str(self.image_path),
                "width_px": self.width_px,
                "height_px": self.height_px,
                "zoom_levels": [str(path) for path in self.zoom_paths],
            },
            "positions": [position.as_leaflet_point() for position in self.positions],
        }
