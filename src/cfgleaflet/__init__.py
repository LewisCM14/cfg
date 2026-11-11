"""Utilities for locating fitting identifiers on CFG images.

This package reads a image and a text list of fitting identifiers,
searches for each identifier in the rendered drawing, and emits Leaflet-ready
coordinates plus a rasterized zoom pyramid for map-based interaction.
"""

from .models import FittingIdentifier, ProcessingResult
from .processing import (
    detect_identifier_positions,
    load_cfg_image,
    parse_fitting_file,
    process_image,
)

__all__ = [
    "FittingIdentifier",
    "ProcessingResult",
    "detect_identifier_positions",
    "load_cfg_image",
    "parse_fitting_file",
    "process_image",
]
