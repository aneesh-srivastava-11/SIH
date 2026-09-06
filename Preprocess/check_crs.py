import rasterio

files = {
    # Replace both paths below with the REAL full paths to your files.
    # Easiest way to get them right: in File Explorer, right-click each
    # .tif -> Copy as path, then paste here (keep the leading r before
    # the quotes, and keep the r"..." wrapper -- that avoids backslash
    # escape issues on Windows paths).
    "WAC": r"C:\Users\ANEESH\Desktop\SIH\data\LunarLROLROC-WAC_MAP2_SIMP.tif",
    "NAC": r"C:\Users\ANEESH\Desktop\SIH\data\M1529798616LC.IMG",
    "IIR": r"C:\Users\ANEESH\Desktop\SIH\data\ch2_iir_nri_20220221T1109265965_d_img_d18.qub",
    "OHR": r"C:\Users\ANEESH\Desktop\SIH\data\ch2_ohr_ncp_20210401T2357376656_d_img_d18.img",
}

for name, path in files.items():
    with rasterio.open(path) as src:
        print(f"{name}: {path}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")
        print()