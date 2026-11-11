from __future__ import annotations

import argparse
from pathlib import Path

from .processing import process_image


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for image processing.

    Returns:
        An ``ArgumentParser`` configured for the image-processing command.
    """
    parser = argparse.ArgumentParser(
        description="Locate fitting identifiers in a CFG images.",
    )
    parser.add_argument("cfg_image", type=Path, help="Path to the CFG image")
    parser.add_argument(
        "fitting_list",
        type=Path,
        help="Text file listing fittings (one per line or comma-separated)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON and rasterized files",
    )
    return parser


def main() -> None:
    """Execute the command-line processing workflow."""
    parser = build_parser()
    args = parser.parse_args()
    process_image(args.cfg_image, args.fitting_list, args.output_dir)


if __name__ == "__main__":
    main()
