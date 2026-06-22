#!/usr/bin/env python3
"""
build_finetune_zip.py - Rebuild finetune_data.zip from your labeled clips.

Run this from the "Line Calls" folder, in the same Python env you use for
labeling (the one with opencv):

    python build_finetune_zip.py

It packages each (video, CSV) pair below into the TrackNetV3 dataset layout
the Colab notebook expects, then zips it to finetune_data.zip (backing up any
existing zip to finetune_data.zip.bak first).

The notebook's data-prep step holds out the LAST clip alphabetically as the
validation set. With the two clips below that means:
    train -> Line-training-1   (1940 frames)
    val   -> line-training-6   (937 frames)

To add more clips later, just append (video, csv) tuples to CLIPS and re-run.
"""

import os
import shutil
import sys
import tempfile
import zipfile

import cv2  # already in your labeling env

HERE = os.path.dirname(os.path.abspath(__file__))

# (video file, label CSV) pairs to include. Add more as you label more clips.
# NOTE: line-training-6 was pillarboxed (portrait content inside a 1920x1080
# landscape frame with black bars). We use the de-pillarboxed 608x1080 crop
# (labels shifted by -656) so it matches Line-training-1's portrait framing.
CLIPS = [
    ("labeled_data/Line-training-1/Line-training-1.mp4",
     "labeled_data/Line-training-1/Line-training-1_ball.csv"),
    ("labeled_data/line-training-6_cropped/line-training-6_cropped.mp4",
     "labeled_data/line-training-6_cropped/line-training-6_cropped_ball.csv"),
]

MATCH = "match24"
OUT_ZIP = os.path.join(HERE, "finetune_data.zip")


def count_csv_rows(path):
    with open(path, "r", newline="") as fh:
        return sum(1 for _ in fh) - 1  # minus header


def main():
    # finetune.py (your customized trainer): prefer the copy already in the
    # existing zip; fall back to the repo checkout.
    finetune_src = None
    if os.path.exists(OUT_ZIP):
        with zipfile.ZipFile(OUT_ZIP) as z:
            if "finetune.py" in z.namelist():
                finetune_src = ("zip", OUT_ZIP)
    if finetune_src is None:
        repo_ft = os.path.join(HERE, "TrackNetV3", "finetune.py")
        if os.path.exists(repo_ft):
            finetune_src = ("file", repo_ft)
    if finetune_src is None:
        sys.exit("ERROR: could not find finetune.py (in finetune_data.zip or TrackNetV3/).")

    build = tempfile.mkdtemp(prefix="ftz_")
    tm = os.path.join(build, "data", "train", MATCH)
    os.makedirs(os.path.join(tm, "video"))
    os.makedirs(os.path.join(tm, "csv"))

    # 1. finetune.py at the zip root
    if finetune_src[0] == "zip":
        with zipfile.ZipFile(finetune_src[1]) as z, \
             z.open("finetune.py") as src, \
             open(os.path.join(build, "finetune.py"), "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy(finetune_src[1], os.path.join(build, "finetune.py"))

    # 2. validate + copy each clip
    problems = []
    for video, csvf in CLIPS:
        vp = os.path.join(HERE, video)
        cp = os.path.join(HERE, csvf)
        if not os.path.exists(vp):
            sys.exit(f"ERROR: missing video {vp}")
        if not os.path.exists(cp):
            sys.exit(f"ERROR: missing CSV {cp}")

        cap = cv2.VideoCapture(vp)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_rows = count_csv_rows(cp)
        name = os.path.splitext(os.path.basename(video))[0]

        if n_frames != n_rows:
            problems.append(
                f"  {name}: video has {n_frames} frames but CSV has {n_rows} label rows"
            )
        print(f"  {name}: {n_rows} labeled frames  (video frames: {n_frames})")

        shutil.copy(vp, os.path.join(tm, "video", f"{name}.mp4"))
        shutil.copy(cp, os.path.join(tm, "csv", f"{name}_ball.csv"))

    if problems:
        print("\nABORTING - frame/label count mismatch (would corrupt training):")
        print("\n".join(problems))
        print("Re-export the CSV(s) so every video frame has exactly one row.")
        shutil.rmtree(build, ignore_errors=True)
        sys.exit(1)

    # 3. back up old zip, then write the new one
    if os.path.exists(OUT_ZIP):
        shutil.copy(OUT_ZIP, OUT_ZIP + ".bak")
        print(f"\n  backed up existing zip -> {os.path.basename(OUT_ZIP)}.bak")

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(build):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, build))

    shutil.rmtree(build, ignore_errors=True)
    print(f"\nDone -> {OUT_ZIP}")
    print("Upload this file in the Colab notebook (cell 4) and run the rest top to bottom.")


if __name__ == "__main__":
    main()
