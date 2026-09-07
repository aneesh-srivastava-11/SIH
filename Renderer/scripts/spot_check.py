import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import numpy as np
from PIL import Image

def main():
    out_dir = "dataset_output/probe_set"
    
    a_files = sorted(glob.glob(os.path.join(out_dir, "*_a.png")))
    npz_files = sorted(glob.glob(os.path.join(out_dir, "*_meta.npz")))
    
    if len(a_files) == 0:
        print("No pairs found!")
        return
        
    n_show = min(5, len(a_files))
    
    # 5 rows, 3 columns
    sample_a = Image.open(a_files[0])
    w, h = sample_a.size
    
    canvas = Image.new('L', (w * 3, h * n_show))
    
    for i in range(n_show):
        img_a = Image.open(a_files[i])
        b_file = a_files[i].replace("_a.png", "_b.png")
        img_b = Image.open(b_file)
        
        meta = np.load(npz_files[i])
        mask = meta["match_mask"]
        img_mask = Image.fromarray((mask * 255).astype(np.uint8))
        
        canvas.paste(img_a, (0, i * h))
        canvas.paste(img_b, (w, i * h))
        canvas.paste(img_mask, (w * 2, i * h))
        
    canvas.save("spot_check_output.png")
    print("Saved spot_check_output.png")

if __name__ == "__main__":
    main()
