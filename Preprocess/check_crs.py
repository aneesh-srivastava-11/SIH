import rasterio

files = {
    # Replace both paths below with the REAL full paths to your files.
    # Easiest way to get them right: in File Explorer, right-click each
    # .tif -> Copy as path, then paste here (keep the leading r before
    # the quotes, and keep the r"..." wrapper -- that avoids backslash
    # escape issues on Windows paths).
    "WAC": r"C:\Users\kushg\Downloads\SIH\Preprocess\data\LunarLROLROC-WAC_MAP2_SIMP.tif",
    "NAC": r"C:\Users\kushg\Downloads\SIH\Preprocess\data\M1529798616LC.IMG",
}

for name, path in files.items():
    with rasterio.open(path) as src:
        print(f"{name}: {path}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print()