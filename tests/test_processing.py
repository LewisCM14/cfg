from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from cfgleaflet import __all__
from cfgleaflet.cli import build_parser
from cfgleaflet.cli import main as cli_main
from cfgleaflet.models import FittingIdentifier
from cfgleaflet.processing import (
    _pick_font,
    create_zoom_levels,
    detect_identifier_positions,
    load_cfg_image,
    parse_fitting_file,
    process_image,
)


def _make_image(path: str | Path, *, mode: str = "RGB") -> None:
    image = Image.new(mode, (500, 300), color="white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=32)
    draw.text((80, 50), "A", fill="black", font=font)
    draw.text((220, 90), "B", fill="black", font=font)
    draw.text((360, 170), "C", fill="black", font=font)
    image.save(path, format="PNG")


def test_parse_fitting_file(tmp_path: Path) -> None:
    fitting_path = tmp_path / "positions.txt"
    fitting_path.write_text("A\nB, C\nD\n", encoding="utf-8")

    assert parse_fitting_file(fitting_path) == ["A", "B", "C", "D"]


def test_detect_identifier_positions_and_zoom_levels(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.cfg"
    _make_image(image_path)

    with Image.open(image_path) as image:
        detected = detect_identifier_positions(image, ["A", "B", "C"])

    assert {entry.fitting_id for entry in detected} == {"A", "B", "C"}
    assert next(entry for entry in detected if entry.fitting_id == "A").x > 0
    assert next(entry for entry in detected if entry.fitting_id == "A").y > 0

    with Image.open(image_path) as image:
        zoom_dir = tmp_path / "tiles"
        zoom_paths = create_zoom_levels(image, zoom_dir)

    assert len(zoom_paths) == 5
    assert all(path.exists() for path in zoom_paths)


def test_load_cfg_image_handles_grayscale_input(tmp_path: Path) -> None:
    image_path = tmp_path / "monochrome.cfg"
    _make_image(image_path, mode="L")

    image = load_cfg_image(image_path)
    assert image.mode == "RGB"
    assert image.size == (500, 300)


def test_build_parser_and_cli_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "sample.cfg"
    fitting_path = tmp_path / "positions.txt"
    output_dir = tmp_path / "output"

    _make_image(image_path)
    fitting_path.write_text("A\nB\nC\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        [str(image_path), str(fitting_path), "--output-dir", str(output_dir)],
    )
    assert args.output_dir == output_dir

    monkeypatch.setattr(
        sys,
        "argv",
        ["cfgleaflet", str(image_path), str(fitting_path), "--output-dir", str(output_dir)],
    )
    cli_main()

    assert (output_dir / "leaflet-positions.json").exists()


def test_module_entrypoint_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = tmp_path / "sample.cfg"
    fitting_path = tmp_path / "positions.txt"
    output_dir = tmp_path / "output"

    _make_image(image_path)
    fitting_path.write_text("A\nB\nC\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["cfgleaflet", str(image_path), str(fitting_path), "--output-dir", str(output_dir)],
    )

    sys.modules.pop("cfgleaflet.cli", None)
    runpy.run_module("cfgleaflet.__main__", run_name="__main__")

    assert (output_dir / "leaflet-positions.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        ["cfgleaflet", str(image_path), str(fitting_path), "--output-dir", str(output_dir)],
    )
    sys.modules.pop("cfgleaflet.cli", None)
    runpy.run_module("cfgleaflet.cli", run_name="__main__")


def test_process_image_writes_json_and_result(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.cfg"
    fitting_path = tmp_path / "positions.txt"
    output_dir = tmp_path / "output"

    _make_image(image_path)
    fitting_path.write_text("A\nB\nC\n", encoding="utf-8")

    result = process_image(image_path, fitting_path, output_dir)

    assert result.width_px == 500
    assert result.height_px == 300
    assert len(result.positions) == 3
    payload_path = output_dir / "leaflet-positions.json"
    assert payload_path.exists()

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["image"]["width_px"] == 500
    assert payload["image"]["height_px"] == 300
    assert len(payload["positions"]) == 3
    assert {entry["id"] for entry in payload["positions"]} == {"A", "B", "C"}
    assert len(payload["image"]["zoom_levels"]) == 5
    assert (output_dir / "hitboxes.json").exists()
    assert (output_dir / "raster").is_dir()
    assert len(list((output_dir / "raster").glob("zoom_*.png"))) == 5
    assert result.hitboxes_path is not None and result.hitboxes_path.exists()
    assert result.svg_path is not None and result.svg_path.exists()
    assert result.to_leaflet_payload()["image"]["width_px"] == 500
    assert result.to_leaflet_payload()["positions"][0]["id"] in {"A", "B", "C"}
    assert len(payload["hitboxes"]) == 3

    location = FittingIdentifier(fitting_id="A", x=10.0, y=20.0, radius_px=5.0)
    assert location.as_leaflet_point() == {"id": "A", "x": 10.0, "y": 20.0, "radius_px": 5.0}
    assert FittingIdentifier(fitting_id="B", x=1.0, y=2.0).as_leaflet_point()["id"] == "B"
    assert "detect_identifier_positions" in __all__


def test_detect_identifier_positions_skips_duplicates(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.cfg"
    _make_image(image_path)

    with Image.open(image_path) as image:
        detected = detect_identifier_positions(image, ["A", "A"])

    assert len(detected) == 1
    assert all(entry.fitting_id == "A" for entry in detected)


def test_detect_identifier_positions_preserves_multiple_occurrences(tmp_path: Path) -> None:
    image = Image.new("RGB", (500, 200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=32)
    draw.text((40, 40), "A", fill="black", font=font)
    draw.text((230, 90), "A", fill="black", font=font)

    detected = detect_identifier_positions(image, ["A"])
    assert len(detected) == 2
    assert {entry.fitting_id for entry in detected} == {"A"}
    xs = sorted(entry.x for entry in detected)
    assert xs[1] - xs[0] > 100.0


def test_process_image_rejects_missing_ids(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.cfg"
    fitting_path = tmp_path / "positions.txt"
    output_dir = tmp_path / "output"

    _make_image(image_path)
    fitting_path.write_text("X\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to locate fitting identifier 'X'"):
        process_image(image_path, fitting_path, output_dir)


def test_missing_paths_raise_file_not_found(tmp_path: Path) -> None:
    missing_image = tmp_path / "missing.cfg"
    missing_list = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_cfg_image(missing_image)

    with pytest.raises(FileNotFoundError):
        parse_fitting_file(missing_list)


def test_empty_fitting_list_and_custom_zoom_levels(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.cfg"
    fitting_path = tmp_path / "positions.txt"
    output_dir = tmp_path / "output"
    _make_image(image_path)
    fitting_path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError):
        process_image(image_path, fitting_path, output_dir)

    zoom_dir = tmp_path / "custom-zoom"
    with Image.open(image_path) as image:
        paths = create_zoom_levels(image, zoom_dir, scales=(0.5, 1.0))
    assert len(paths) == 2


def test_pick_font_returns_default_when_system_fonts_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if str(path).startswith("/usr/share/fonts"):
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", fake_exists)
    font = _pick_font(24)
    assert font is not None


def test_internal_helpers_and_font_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)
    font = _pick_font(16)
    assert font is not None

    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert image.size == (10, 10)

    # direct helper path, not an actual file, but confirms logic is callable
    assert _pick_font(12) is not None
