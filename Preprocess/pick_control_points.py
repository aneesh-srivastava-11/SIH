"""
Manually pick corresponding control points between two images.

Shows both images side by side. Click a feature (e.g. a crater center)
in the LEFT image, then click the SAME feature in the RIGHT image.
Repeat 15-25 times (project doc recommends this range -- doc section
9.2, method 3). Points are saved in matched left/right pairs.

Controls:
    Left-click on LEFT image  -> record point in img1
    Left-click on RIGHT image -> record point in img2 (must alternate:
                                   img1 point, then img2 point, etc.)
    'u' key -> undo the last recorded point
    'q' key or close window -> save and quit

Usage:
    python pick_control_points.py --img1 ..\\basebenchmarking\\data\\raw\\p1_ohrc.npy \\
                                   --img2 ..\\basebenchmarking\\data\\raw\\p1_nac.npy \\
                                   --out ..\\basebenchmarking\\data\\ground_truth\\p1_control_points.json

IMPORTANT: for large images (e.g. full OHRC crop), this downsamples for
DISPLAY only -- clicked coordinates are automatically rescaled back to
full-resolution pixel coordinates before saving, so the saved control
points are always in the original image's pixel space.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class ControlPointSession:
    """
    Holds all the click/undo/rescale logic independent of matplotlib's
    event loop, so it can be unit-tested with fake events (no display
    needed) before trusting it in an interactive session.
    """
    def __init__(self, step1, step2):
        self.step1 = step1
        self.step2 = step2
        self.points = {"img1": [], "img2": []}

    def click(self, which, disp_x, disp_y):
        """which is 'img1' or 'img2'. Returns True if the click was accepted."""
        if which == "img1":
            if len(self.points["img1"]) != len(self.points["img2"]):
                return False  # must click img1 first, before a matching img2 click
            self.points["img1"].append([disp_x * self.step1, disp_y * self.step1])
            return True
        elif which == "img2":
            if len(self.points["img2"]) != len(self.points["img1"]) - 1:
                return False  # need exactly one pending img1 click first
            self.points["img2"].append([disp_x * self.step2, disp_y * self.step2])
            return True
        return False

    def undo(self):
        """Remove the most recent point (or pending pair). Returns what was removed."""
        if len(self.points["img2"]) < len(self.points["img1"]):
            return "img1", self.points["img1"].pop()
        elif self.points["img1"] and self.points["img2"]:
            p1 = self.points["img1"].pop()
            p2 = self.points["img2"].pop()
            return "pair", (p1, p2)
        return None, None

    def complete_pairs(self):
        n = min(len(self.points["img1"]), len(self.points["img2"]))
        return self.points["img1"][:n], self.points["img2"][:n]


def _run_unit_tests():
    """Sanity-check ControlPointSession logic with fake events -- no display needed."""
    s = ControlPointSession(step1=4, step2=2)

    assert s.click("img2", 10, 10) is False, "should reject img2 click before any img1 click"
    assert s.click("img1", 10, 20) is True
    assert s.points["img1"] == [[40, 80]], f"got {s.points['img1']}"  # scaled by step1=4
    assert s.click("img1", 5, 5) is False, "should reject second img1 click before matching img2"
    assert s.click("img2", 3, 6) is True
    assert s.points["img2"] == [[6, 12]], f"got {s.points['img2']}"  # scaled by step2=2

    # second complete pair
    assert s.click("img1", 1, 1) is True
    assert s.click("img2", 2, 2) is True
    p1, p2 = s.complete_pairs()
    assert len(p1) == 2 and len(p2) == 2

    # undo the completed pair
    kind, removed = s.undo()
    assert kind == "pair"
    p1, p2 = s.complete_pairs()
    assert len(p1) == 1 and len(p2) == 1

    # undo a lone pending img1 click
    s.click("img1", 9, 9)
    kind, removed = s.undo()
    assert kind == "img1"
    p1, p2 = s.complete_pairs()
    assert len(p1) == 1 and len(p2) == 1  # back to the one completed pair

    print("All ControlPointSession unit tests passed.")


def normalize_for_display(arr, pct_clip=1.0):
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, pct_clip), np.percentile(arr, 100 - pct_clip)
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def downsample_for_display(arr, max_dim=1200):
    h, w = arr.shape
    step = max(1, max(h, w) // max_dim)
    return arr[::step, ::step], step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1")
    ap.add_argument("--img2")
    ap.add_argument("--out")
    ap.add_argument("--img1-band", type=int, default=None,
                     help="If img1 is a 3D cube, extract this band first")
    ap.add_argument("--img2-band", type=int, default=None)
    ap.add_argument("--test", action="store_true",
                     help="Run internal unit tests and exit (no display needed)")
    args = ap.parse_args()

    if args.test:
        _run_unit_tests()
        return

    if not (args.img1 and args.img2 and args.out):
        ap.error("--img1, --img2, and --out are required unless --test is given")

    img1 = np.load(args.img1)
    img2 = np.load(args.img2)
    if args.img1_band is not None:
        img1 = img1[args.img1_band]
    if args.img2_band is not None:
        img2 = img2[args.img2_band]

    img1_disp, step1 = downsample_for_display(normalize_for_display(img1))
    img2_disp, step2 = downsample_for_display(normalize_for_display(img2))

    session = ControlPointSession(step1, step2)
    markers = {"img1": [], "img2": []}

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(img1_disp, cmap="gray")
    axes[0].set_title(f"img1 (click here first each pair) -- {img1.shape}")
    axes[1].imshow(img2_disp, cmap="gray")
    axes[1].set_title(f"img2 (click matching feature here) -- {img2.shape}")
    for ax in axes:
        ax.axis("off")
    status_text = fig.text(0.5, 0.02, "", ha="center", fontsize=11)

    def update_status():
        p1, p2 = session.complete_pairs()
        n1, n2 = len(session.points["img1"]), len(session.points["img2"])
        if n1 == n2:
            status_text.set_text(
                f"{len(p1)} complete pairs. Click a feature in img1 (left) next. "
                f"Press 'u' to undo, 'q' to save & quit."
            )
        else:
            status_text.set_text(
                f"{len(p1)} complete pairs, waiting for matching click in img2 (right). "
                f"Press 'u' to undo."
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == axes[0]:
            if session.click("img1", event.xdata, event.ydata):
                m, = axes[0].plot(event.xdata, event.ydata, "r+", markersize=12, markeredgewidth=2)
                markers["img1"].append(m)
        elif event.inaxes == axes[1]:
            if session.click("img2", event.xdata, event.ydata):
                m, = axes[1].plot(event.xdata, event.ydata, "r+", markersize=12, markeredgewidth=2)
                markers["img2"].append(m)
        update_status()

    def on_key(event):
        if event.key == "u":
            kind, _ = session.undo()
            if kind == "img1":
                markers["img1"].pop().remove()
            elif kind == "pair":
                markers["img1"].pop().remove()
                markers["img2"].pop().remove()
            update_status()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_status()
    plt.tight_layout()
    plt.show()

    img1_pts, img2_pts = session.complete_pairs()
    n_pairs = len(img1_pts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "img1_path": args.img1,
            "img2_path": args.img2,
            "n_points": n_pairs,
            "control_points": [
                {"img1_xy": img1_pts[i], "img2_xy": img2_pts[i]}
                for i in range(n_pairs)
            ],
            "usage_note": (
                "Coordinates are (x, y) = (col, row) in FULL RESOLUTION pixel "
                "space of the original .npy files. These points must be held "
                "OUT of any transform fitting and used only for scoring "
                "accuracy afterward (project doc section 9.1's core warning)."
            ),
        }, f, indent=2)

    print(f"\nSaved {n_pairs} control point pairs to: {out_path}")
    if n_pairs < 15:
        print(f"NOTE: doc recommends 15-25 points per pair; you have {n_pairs}. "
              "Consider running again to add more before treating this as final.")


if __name__ == "__main__":
    main()