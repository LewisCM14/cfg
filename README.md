# CFGLeaflet

CFGLeaflet reads a CFG image and a text list of position identifiers, locates each position, and outputs a Leaflet-friendly JSON payload plus a five-level raster pyramid.

## Features

- Reads raster images stored as `.cfg` or standard image files (PNG/JPEG/TIFF/etc.)
- Parses ID lists from `.txt` files
- Locates identifiers in the image by template matching
- Produces a JSON structure suitable for Leaflet image overlays and interactions
- Rasters the source image into five zoom levels for overlay rendering
- Validated with Ruff, Mypy, pytest, and 100% coverage

## Quick start

```bash
uv sync --group dev
uv run cfgleaflet process ./image.cfg ./positions.txt --output-dir ./output
```

The command creates:

- `output/leaflet-positions.json`
- `output/zoom_0.png` through `output/zoom_4.png`
