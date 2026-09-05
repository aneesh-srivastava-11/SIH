"""
Build a corners.json for the NAC product, using the corner lat/lon values
already read off its LROC detail page (NAC CDR has no embedded map
projection, so there's nothing to extract programmatically the way
GeoTIFFs have -- these values were manually copied from the LROC website).

Run once to produce data/nac_geometry.json, then pass that file to
crop_to_overlap.py the same way you pass the PDS4 parser's geometry.json
for OHRC/IIRS.
"""

import json
from pathlib import Path

geometry = {
    "product_id": "M1529798616LE",
    "instrument": "LROC NAC",
    "corners": {
        "upper_left": ["-13.86", "25.05"],
        "upper_right": ["-13.87", "25.17"],
        "lower_right": ["-12.58", "25.31"],
        "lower_left": ["-12.56", "25.19"],
    },
}

out_path = Path("data") / "nac_geometry.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(geometry, f, indent=2)

print(f"Wrote {out_path}")
print(json.dumps(geometry, indent=2))