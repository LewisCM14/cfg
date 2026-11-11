from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .models import FittingIdentifier, ProcessingResult

DEFAULT_ZOOM_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0)


def load_cfg_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and normalize it to an RGB raster.

    Args:
        image_path: A filesystem path to the source image.

    Returns:
        A Pillow image converted to ``RGB`` for consistent processing.

    Raises:
        FileNotFoundError: If the supplied path does not exist.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as source:
        image = source.copy()

    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def parse_fitting_file(fitting_list_path: str | Path) -> list[str]:
    """Parse a text file containing fitting IDs.

    The source file may contain one ID per line, comma-separated IDs, or both.

    Args:
        fitting_list_path: The path to a plain-text list of fitting identifiers.

    Returns:
        A list of fitting IDs normalized to uppercase.

    Raises:
        FileNotFoundError: If the fitting list does not exist.
    """
    path = Path(fitting_list_path)
    if not path.exists():
        raise FileNotFoundError(f"Fitting list not found: {path}")

    identifiers: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        for token in raw_line.replace(";", ",").split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            identifiers.append(cleaned.upper())
    return identifiers


def detect_identifier_positions(
    image: Image.Image,
    fitting_ids: Sequence[str],
    *,
    font_size_min: int = 18,
    font_size_max: int = 160,
    step: int = 4,
) -> list[FittingIdentifier]:
    """Locate each identifier in an image.

    Multiple copies of the same identifier are preserved as multiple
    ``FittingIdentifier`` entries in the returned list.

    Args:
        image: A Pillow image containing the image.
        fitting_ids: The list of fitting identifiers to search for.
        font_size_min: Lower bound for font-size matching.
        font_size_max: Upper bound for font-size matching.
        step: The scan step between font sizes.

    Returns:
        A list of fitting identifiers and their source-image coordinates.

    Raises:
        ValueError: If any identifier cannot be found.
    """
    grayscale = image.convert("L")
    gray_array = np.asarray(grayscale, dtype=np.float32)
    detected: list[FittingIdentifier] = []

    for fitting_id in fitting_ids:
        locations = _locate_texts(gray_array, fitting_id, font_size_min, font_size_max, step)
        if not locations:
            raise ValueError(f"Unable to locate fitting identifier {fitting_id!r} in the image.")

        for x, y in locations:
            if any(
                existing.fitting_id == fitting_id
                and abs(existing.x - x) < 5.0
                and abs(existing.y - y) < 5.0
                for existing in detected
            ):
                continue
            detected.append(
                FittingIdentifier(
                    fitting_id=fitting_id,
                    x=float(x),
                    y=float(y),
                    radius_px=12.0,
                )
            )

    return detected


def create_zoom_levels(
    image: Image.Image,
    output_dir: str | Path,
    *,
    scales: Iterable[float] = DEFAULT_ZOOM_SCALES,
) -> list[Path]:
    """Create a rasterized zoom pyramid for the source image.

    Args:
        image: The source image to rasterize.
        output_dir: Destination directory for the generated zoom layers.
        scales: Multipliers applied to the source image dimensions.

    Returns:
        A list of paths to the generated zoom PNG files.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for index, scale in enumerate(tuple(scales)):
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        raster = image.resize((width, height), Image.Resampling.LANCZOS)
        output_path = target_dir / f"zoom_{index}.png"
        raster.save(output_path)
        generated_paths.append(output_path)
    return generated_paths


def process_image(
    cfg_image_path: str | Path,
    fitting_list_path: str | Path,
    output_dir: str | Path,
) -> ProcessingResult:
    """Process a image image and fitting list into Leaflet-ready outputs.

    A vector-first workflow is approximated here by identifying labels in the
    rendered raster, creating a scalar SVG copy of the image, and then
    rasterizing that SVG for Leaflet tile use.

    Args:
        cfg_image_path: Path to the CFG image.
        fitting_list_path: Path to the fitting identifier list.
        output_dir: Directory that will receive the JSON output and zoom layers.

    Returns:
        A ``ProcessingResult`` describing the generated files and coordinates.

    Raises:
        ValueError: If the fitting list is empty or any fitting cannot be found.
    """
    image = load_cfg_image(cfg_image_path)
    fitting_ids = parse_fitting_file(fitting_list_path)
    if not fitting_ids:
        raise ValueError("No fitting identifiers were found in the fitting list.")

    positions = detect_identifier_positions(image, fitting_ids)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    raster_dir = output_root / "raster"
    zoom_paths = create_zoom_levels(image, raster_dir)

    svg_path = output_root / "image-copy.svg"
    source_copy_path = output_root / "image-copy.png"
    image.save(source_copy_path)
    svg_path.write_text(
        _build_svg_copy(source_copy_path, image.width, image.height),
        encoding="utf-8",
    )

    hitboxes_payload = _build_hitboxes_payload(positions)
    hitboxes_path = output_root / "hitboxes.json"
    hitboxes_path.write_text(json.dumps(hitboxes_payload, indent=2), encoding="utf-8")

    payload = {
        "image": {
            "source": str(Path(cfg_image_path)),
            "width_px": image.width,
            "height_px": image.height,
            "zoom_levels": [str(path) for path in zoom_paths],
            "svg_copy": str(svg_path),
        },
        "positions": [position.as_leaflet_point() for position in positions],
        "hitboxes": hitboxes_payload,
    }
    json_output_path = output_root / "leaflet-positions.json"
    json_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return ProcessingResult(
        image_path=Path(cfg_image_path),
        output_dir=output_root,
        width_px=image.width,
        height_px=image.height,
        positions=positions,
        zoom_paths=zoom_paths,
        hitboxes_path=hitboxes_path,
        svg_path=svg_path,
    )


def _locate_texts(
    grayscale: np.ndarray,
    fitting_id: str,
    font_size_min: int,
    font_size_max: int,
    step: int,
) -> list[tuple[float, float]]:
    """Return all locations matching a target label within the rendered image."""
    matches: list[tuple[float, float]] = []

    for font_size in range(font_size_min, font_size_max + 1, step):
        template = _render_label(fitting_id, font_size)
        template_array = np.asarray(template.convert("L"), dtype=np.float32)
        template_mask = template_array < 245

        stride_x = max(1, template_array.shape[1] // 8)
        stride_y = max(1, template_array.shape[0] // 8)
        for y in range(0, grayscale.shape[0] - template_array.shape[0] + 1, stride_y):
            for x in range(0, grayscale.shape[1] - template_array.shape[1] + 1, stride_x):
                patch = grayscale[y : y + template_array.shape[0], x : x + template_array.shape[1]]
                patch_mask = patch < 200
                if not np.any(patch_mask):
                    continue

                overlap = np.logical_and(patch_mask, template_mask).sum()
                similarity = overlap / max(1, np.count_nonzero(template_mask))
                if similarity < 0.75:
                    continue

                if any(
                    abs(existing_x - x) < 20 and abs(existing_y - y) < 20
                    for existing_x, existing_y in matches
                ):
                    continue
                matches.append((float(x), float(y)))

    return matches


def _build_hitboxes_payload(positions: Sequence[FittingIdentifier]) -> list[dict[str, object]]:
    """Convert fitting identifiers into a separate hitbox payload for downstream tools."""
    boxes: list[dict[str, object]] = []
    for location in positions:
        half_size = max(location.radius_px, 12.0) / 2
        boxes.append(
            {
                "id": location.fitting_id,
                "label": location.fitting_id,
                "x": round(location.x - half_size, 2),
                "y": round(location.y - half_size, 2),
                "width": round(max(location.radius_px, 12.0), 2),
                "height": round(max(location.radius_px, 12.0), 2),
            }
        )
    return boxes


def _build_svg_copy(image_path: Path, width: int, height: int) -> str:
    """Create a simple SVG wrapper around a source PNG for the frontend pipeline."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
        f'  <image href="{image_path.name}" x="0" y="0" width="{width}" height="{height}" />\n'
        "</svg>\n"
    )


def _render_label(label: str, size: int) -> Image.Image:
    """Render a label into a grayscale canvas for template matching."""
    font = _pick_font(size)
    width = max(24, int(len(label) * size * 0.7) + 20)
    height = max(24, size + 20)
    canvas = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), label, font=font, fill=0)
    return canvas


def _pick_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Resolve a font object using the system font library when available."""
    candidate_paths = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    )
    for path in candidate_paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()
