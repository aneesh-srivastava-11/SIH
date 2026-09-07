"""
CRS (Coordinate Reference System) Inspector for Lunar Data Files.

Scans a data directory for supported raster files (.tif, .IMG, .qub, .img)
and prints their CRS, bounds, and resolution information.

Usage:
    python check_crs.py --data-dir ../data
    python check_crs.py --data-dir /path/to/files
"""

import argparse
from pathlib import Path

# Portable default: data/ directory at the project root (one level above Preprocess/)
_DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

# Known raster extensions to look for
_RASTER_EXTENSIONS = {".tif", ".tiff", ".img", ".qub"}


def inspect_crs(data_dir: str) -> None:
    """Discover and inspect CRS for all raster files in the given directory."""
    try:
        import rasterio
    except ImportError:
        print("Error: 'rasterio' is required to inspect CRS. Install it via 'pip install rasterio'.")
        return

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        return

    raster_files = sorted(
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in _RASTER_EXTENSIONS
    )

    if not raster_files:
        print(f"No raster files found in: {data_path}")
        print(f"Supported extensions: {', '.join(_RASTER_EXTENSIONS)}")
        return

    for filepath in raster_files:
        try:
            with rasterio.open(str(filepath)) as src:
                print(f"{filepath.stem} ({filepath.suffix}):")
                print(f"  Path:   {filepath}")
                print(f"  CRS:    {src.crs}")
                print(f"  Bounds: {src.bounds}")
                print(f"  Size:   {src.width} x {src.height}")
                print()
        except Exception as e:
            print(f"{filepath.name}: ERROR - {e}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect CRS and bounds of raster files in a directory."
    )
    parser.add_argument(
        "--data-dir",
        default=_DEFAULT_DATA_DIR,
        help=f"Directory containing raster files (default: {_DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    inspect_crs(args.data_dir)