"""
PDS4 reader/parser for Chandrayaan-2 products (OHRC, IIRS, etc.)

Usage:
    python pds4_parser.py path/to/product.xml

Requires:
    pip install pds4_tools --break-system-packages

Notes:
- Point this at the .xml LABEL file, not the .IMG — pds4_tools reads the
  label and uses it to correctly interpret the paired .IMG data file
  (which must sit in the same folder, matching filename).
- Prints out the metadata every downstream step (matching, rendering,
  evaluation) depends on: footprint corners, GSD, sun geometry.
- Also saves a cropped-preview-ready numpy array + a small JSON sidecar
  with the extracted geometry, so you don't have to re-parse the XML
  every time you touch the image.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pds4_tools


def load_product(xml_path: str):
    """Load a PDS4 product given its .xml label path."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Label file not found: {xml_path}")

    structures = pds4_tools.read(str(xml_path), quiet=True)
    return structures


def extract_image_array(structures):
    """
    Pull the first image/array data structure out of the PDS4 product.
    Chandrayaan-2 OHRC/IIRS products typically have one primary array.
    """
    for struct in structures:
        # pds4_tools Structure objects expose .data for arrays/images
        if hasattr(struct, "data") and struct.data is not None:
            arr = np.asarray(struct.data)
            return arr, struct
    raise ValueError("No image/array data structure found in this product.")


def extract_geometry(structures):
    """
    Walk the PDS4 label XML for the fields we actually need downstream:
    footprint corners, ground sample distance, sun angles, view angles.

    PDS4 labels vary by mission/instrument in exactly which XML tags carry
    this info, so this pulls from label_meta_data (the parsed label dict)
    and falls back gracefully if a field isn't present. INSPECT THE PRINTED
    OUTPUT the first time you run this on a new product type (OHRC vs IIRS
    labels differ) and adjust the key paths below if something prints None.
    """
    geometry = {
        "product_id": None,
        "instrument": None,
        "start_time": None,
        "corners": None,          # geographic footprint corners
        "gsd_m": None,             # ground sample distance, meters/pixel
        "sun_azimuth_deg": None,
        "sun_elevation_deg": None,
        "incidence_angle_deg": None,
        "emission_angle_deg": None,
        "phase_angle_deg": None,
    }

    label = structures.label  # raw PDS4 label object (lxml-backed)

    def find_text(xpath_local_name):
        """
        Search the label for the first element whose local tag name
        matches xpath_local_name, ignoring XML namespace prefixes
        (PDS4 labels are namespace-heavy).
        """
        for elem in label.iter():
            tag = elem.tag.split("}")[-1]  # strip {namespace}
            if tag == xpath_local_name and elem.text and elem.text.strip():
                return elem.text.strip()
        return None

    geometry["product_id"] = find_text("logical_identifier")
    geometry["start_time"] = find_text("start_date_time")

    # ISRO's flat CH2 label schema has no dedicated instrument_name tag.
    # "title" carries it as free text, e.g. "Chandrayaan-2 Orbiter OHRC
    # Experiment" — good enough to identify the instrument at a glance.
    geometry["instrument"] = find_text("title")

    # Confirmed real tag names from CH2 OHRC calibrated-product labels
    # (verified against an actual PRADAN download — see project notes).
    # NOTE: emission_angle / phase_angle are genuinely absent from OHRC
    # labels — don't expect them to populate for this product type.
    geometry["sun_azimuth_deg"] = find_text("sun_azimuth")
    geometry["sun_elevation_deg"] = find_text("sun_elevation")
    geometry["incidence_angle_deg"] = find_text("solar_incidence")
    geometry["emission_angle_deg"] = find_text("emission_angle")
    geometry["phase_angle_deg"] = find_text("phase_angle")

    # Ground sample distance / resolution — ISRO calls this pixel_resolution
    geometry["gsd_m"] = find_text("pixel_resolution")

    # Footprint corners — ISRO labels store four explicit named corners
    # rather than a generic repeated lat/lon loop.
    corner_names = ["upper_left", "upper_right", "lower_left", "lower_right"]
    corners = {}
    for corner in corner_names:
        lat = find_text(f"{corner}_latitude")
        lon = find_text(f"{corner}_longitude")
        if lat and lon:
            corners[corner] = (lat, lon)
    geometry["corners"] = corners if corners else None

    return geometry


def main():
    if len(sys.argv) != 2:
        print("Usage: python pds4_parser.py path/to/product.xml")
        sys.exit(1)

    xml_path = sys.argv[1]
    print(f"Loading: {xml_path}")

    structures = load_product(xml_path)

    arr, img_struct = extract_image_array(structures)
    print(f"\nImage array shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"Min/Max pixel value: {arr.min()} / {arr.max()}")

    geometry = extract_geometry(structures)
    print("\nExtracted geometry:")
    print(json.dumps(geometry, indent=2))

    # Save outputs alongside the input, so you have them cached
    out_dir = Path(xml_path).parent
    stem = Path(xml_path).stem

    npy_path = out_dir / f"{stem}_array.npy"
    np.save(npy_path, arr)
    print(f"\nSaved image array to: {npy_path}")

    json_path = out_dir / f"{stem}_geometry.json"
    with open(json_path, "w") as f:
        json.dump(geometry, f, indent=2)
    print(f"Saved geometry to: {json_path}")

    # Flag anything that came back empty so you know what to fix
    missing = [k for k, v in geometry.items() if v is None]
    if missing:
        print(f"\n⚠️  Could not auto-extract: {missing}")
        print("   Open the .xml label in a text editor and search for these")
        print("   fields manually — tag names can differ between OHRC and")
        print("   IIRS labels. Update the find_text() calls above once you")
        print("   see the real tag names.")


if __name__ == "__main__":
    main()