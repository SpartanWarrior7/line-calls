#!/usr/bin/env python3
"""
label_ball.py - Minimal ball-labelling tool for TrackNetV3 fine-tuning.

Step through a video clip frame by frame and click the ball. Writes a CSV in
the exact format TrackNetV3 expects:

    Frame, Visibility, X, Y

  - one row per frame (every frame in the clip gets a row)
  - Visibility = 1 and X,Y = ball-centre pixel coords (ORIGINAL resolution) when visible
  - Visibility = 0, X = 0, Y = 0 when the ball is not visible (occluded / off-screen)

TrackNetV3 trains on stacks of CONSECUTIVE frames, so label a whole clip, not
scattered frames. Trim your video to one rally first (e.g.
`ffmpeg -i in.mp4 -ss 12 -to 19 -c:v libx264 -crf 18 rally1.mp4`) and label all
of it.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  # EASIEST: no typing. Double-click "Label Ball.bat", or just run with no args,
  # to open a big-button home screen (pick a video / review, then click the ball).
  python label_ball.py

  # label a clip from scratch:
  python label_ball.py --video rally1.mp4
  python label_ball.py --video rally1.mp4 --scale 0.75 --loupe-zoom 6

  # drop the result straight into a TrackNetV3 dataset:
  python label_ball.py --video rally1.mp4 \
      --dataset-root TrackNetV3/data --split train --match 24

  # REVIEW a CSV the (w.i.p.) model produced, frame by frame, and fix mistakes.
  # A file-picker pops up so you can choose the model CSV (and the video if you
  # don't pass --video). Corrections are saved back over the same CSV.
  python label_ball.py --review
  python label_ball.py --review --video rally1.mp4 --csv rally1_ball.csv

CONTROLS (shown on screen too)
  Left click .... set ball at cursor (Visibility=1) and ADVANCE one frame
  Right click ... set ball without advancing (fine adjust)
  / ............. set ball at cursor without advancing (same as right click)
  Arrow keys .... nudge the current label 1 pixel (micro-adjust)
  c / Enter ..... (review) CONFIRM this frame is correct as-is and advance
  n / Space ..... mark NOT VISIBLE (Visibility=0) and advance
  d ............. next frame          a ............. previous frame
  w / e ......... jump -10 / +10 frames
  u ............. jump to first UNLABELLED frame (REVIEW: first UNREVIEWED frame)
  z ............. clear this frame's label (undo)
  f ............. toggle fullscreen (also: --fullscreen to start that way)
  m ............. open the SETTINGS menu (display mode + export)
  s ............. save now            q / Esc ....... save and quit

SETTINGS MENU (press m)
  A click-driven screen to choose the display mode - Fit to screen, Fullscreen,
  or Actual size (1:1 pixels) - and to Export. "Export labeled CSV" writes every
  frame in TrackNetV3 format to <video name>-labeled.csv next to the video.
  "Directions" opens a help screen explaining every control.

REVIEW MODE
  The model's CSV is loaded as the starting point. Every prediction shows in
  YELLOW until you sign off on it; confirmed/edited frames turn RED. Press
  c (or Enter) to accept a prediction unchanged, or click / nudge / n to fix
  it - any of those also count as "reviewed". `u` jumps to the next frame you
  haven't reviewed yet, and the status bar tracks reviewed N/total.

Progress is auto-saved; re-running on the same --out resumes where you left off.
In review mode the reviewed-so-far set is remembered too, so you can stop and
pick up the verification pass later.

Dependencies: opencv-python, numpy  (both already in the TrackNetV3 env)
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

# cv2 is only needed for frame extraction and the GUI. Import lazily so the
# pure-logic functions below can be imported/tested without a display.
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# Core logic (no GUI) - kept importable and testable.
# ----------------------------------------------------------------------------

IMG_EXT = "jpg"  # cache format for the labelling display only (not training)


def extract_frames(video_path, frames_dir, ext=IMG_EXT):
    """Decode every frame of `video_path` sequentially to `frames_dir`/<i>.<ext>.

    Returns (num_frames, width, height). Skips extraction if the directory
    already holds the expected number of frames. Sequential decoding means
    frame i here == frame i that TrackNetV3's preprocess.py will extract.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for frame extraction.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(frames_dir, exist_ok=True)
    existing = [f for f in os.listdir(frames_dir) if f.endswith(f".{ext}")]
    if reported > 0 and len(existing) == reported:
        cap.release()
        return reported, width, height

    # Re-extract from scratch to guarantee contiguous 0..N-1 indexing.
    for f in existing:
        os.remove(os.path.join(frames_dir, f))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{i}.{ext}"), frame)
        if i % 100 == 0:
            sys.stdout.write(f"\r  extracting frame {i} ...")
            sys.stdout.flush()
        i += 1
    cap.release()
    sys.stdout.write(f"\r  extracted {i} frames.            \n")
    return i, width, height


def save_labels(csv_path, visibility, xs, ys):
    """Write the TrackNetV3 label CSV: Frame, Visibility, X, Y (one row/frame)."""
    n = len(visibility)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        for i in range(n):
            w.writerow([i, int(visibility[i]), int(round(xs[i])), int(round(ys[i]))])


def load_labels(csv_path, n, return_present=False):
    """Load an existing label CSV into arrays of length n. Missing -> zeros.

    With return_present=True, also returns a boolean array marking which frames
    actually had a row in the CSV (used by review mode to know which frames the
    model produced a prediction for).
    """
    vis = np.zeros(n, dtype=int)
    xs = np.zeros(n, dtype=int)
    ys = np.zeros(n, dtype=int)
    present = np.zeros(n, dtype=bool)
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            i = int(row["Frame"])
            if 0 <= i < n:
                vis[i] = int(float(row["Visibility"]))
                xs[i] = int(float(row["X"]))
                ys[i] = int(float(row["Y"]))
                present[i] = True
    if return_present:
        return vis, xs, ys, present
    return vis, xs, ys


def save_progress(path, labelled, reviewed=None):
    data = {"labelled": [int(i) for i in np.where(labelled)[0]]}
    if reviewed is not None:
        data["reviewed"] = [int(i) for i in np.where(reviewed)[0]]
    with open(path, "w") as fh:
        json.dump(data, fh)


def load_progress(path, n, key="labelled"):
    arr = np.zeros(n, dtype=bool)
    if os.path.exists(path):
        with open(path) as fh:
            for i in json.load(fh).get(key, []):
                if 0 <= i < n:
                    arr[i] = True
    return arr


def place_in_dataset(video_path, csv_path, dataset_root, split, match, rally):
    """Copy clip + CSV into a TrackNetV3 dataset layout.

    <root>/<split>/match<match>/video/<rally>.mp4
    <root>/<split>/match<match>/csv/<rally>_ball.csv
    """
    match_dir = os.path.join(dataset_root, split, f"match{match}")
    video_dir = os.path.join(match_dir, "video")
    csv_dir = os.path.join(match_dir, "csv")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    dst_video = os.path.join(video_dir, f"{rally}.mp4")
    dst_csv = os.path.join(csv_dir, f"{rally}_ball.csv")
    shutil.copy(video_path, dst_video)
    shutil.copy(csv_path, dst_csv)
    return dst_video, dst_csv


# ----------------------------------------------------------------------------
# File pickers (used by --review so you can "upload" a model CSV via a dialog)
# ----------------------------------------------------------------------------

def _ask_open_file(title, patterns):
    """Pop a native open-file dialog and return the chosen path (or None).

    Falls back to a console prompt if Tk isn't available (e.g. headless box).
    `patterns` is a list like [("CSV files", "*.csv"), ("All files", "*.*")].
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        try:
            ans = input(f"{title}\n  enter path (blank to cancel): ").strip()
        except EOFError:
            return None
        return ans or None

    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(title=title, filetypes=patterns)
    root.destroy()
    return path or None


def pick_csv_dialog():
    return _ask_open_file("Select the model CSV to review",
                          [("CSV files", "*.csv"), ("All files", "*.*")])


def pick_video_dialog():
    return _ask_open_file(
        "Select the matching video clip",
        [("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All files", "*.*")])


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

BAR_H = 56  # top status-bar height in display pixels

# Arrow-key codes differ by OS / OpenCV backend; cover Windows, Linux/GTK, macOS.
# (All are large, distinct values, so masking with & 0xFF never hits an ASCII key.)
ARROW_LEFT = {2424832, 65361, 63234}
ARROW_UP = {2490368, 65362, 63232}
ARROW_RIGHT = {2555904, 65363, 63235}
ARROW_DOWN = {2621440, 65364, 63233}


def _blend_circle(img, center, radius, color, alpha):
    """Draw a filled circle onto img at the given opacity (alpha 0..1).

    Blends only the small region around the circle so it's cheap to call many
    times per rendered frame (used for the fading ball trail).
    """
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(img.shape[1], cx + radius + 1), min(img.shape[0], cy + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.circle(overlay, (cx - x0, cy - y0), radius, color, -1)
    cv2.addWeighted(overlay, float(alpha), roi, 1.0 - float(alpha), 0, dst=roi)


def get_screen_size(default=(1920, 1080)):
    """Best-effort desktop resolution (width, height) for sizing the window."""
    try:
        import tkinter as tk
        r = tk.Tk()
        r.withdraw()
        w, h = r.winfo_screenwidth(), r.winfo_screenheight()
        r.destroy()
        if w > 0 and h > 0:
            return int(w), int(h)
    except Exception:
        pass
    return default


# Plain-language help shown on the home screen (kid-friendly, no jargon).
WELCOME_HELP = [
    ("h", "The goal"),
    ("t", "You'll watch a tennis video one picture (frame) at a time."),
    ("t", "Your job is to show the computer exactly where the ball is."),
    ("sp", ""),
    ("h", "Labeling a new video"),
    ("t", "Click right on the ball. It marks the spot and jumps to the next picture."),
    ("t", "Can't see the ball? Press  N  (it's hidden or off the screen)."),
    ("t", "Want to go back one picture? Press  A ."),
    ("t", "Need to move the mark a tiny bit? Use the arrow keys."),
    ("sp", ""),
    ("h", "Fixing the computer's guesses"),
    ("t", "The computer's guess shows up YELLOW."),
    ("t", "If the guess is right, press  C  to keep it (it turns RED)."),
    ("t", "If the guess is wrong, just click the correct spot to fix it."),
    ("t", "Press  U  to jump to the next picture that still needs checking."),
    ("sp", ""),
    ("h", "Finishing up"),
    ("t", "Your work saves by itself. Press  Q  any time to save and stop."),
    ("t", "Inside the labeler, press  M  for the full menu and more help."),
]


def run_welcome():
    """Big-button home screen so nobody has to type a command.

    Returns one of "label", "review", or "quit". Falls back to a tiny console
    menu if OpenCV has no display available.
    """
    if cv2 is None:
        print("\n=== Ball Labeler ===")
        print("  1) Label a new video")
        print("  2) Fix the computer's guesses")
        print("  3) Quit")
        try:
            ans = input("Type 1, 2, or 3 and press Enter: ").strip()
        except EOFError:
            return "quit"
        return {"1": "label", "2": "review", "3": "quit"}.get(ans, "quit")

    win = "Ball Labeler"
    W, H = 900, 620
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    FONT, LINE = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    state = {"page": "home", "result": None, "rects": []}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x0, y0, x1, y1, action) in state["rects"]:
            if x0 <= x <= x1 and y0 <= y <= y1:
                action()
                break

    cv2.setMouseCallback(win, on_mouse)

    def choose(mode):
        state["result"] = mode

    def go(page):
        state["page"] = page

    def render_home():
        state["rects"] = []
        canvas = np.full((H, W, 3), 30, np.uint8)
        cv2.rectangle(canvas, (0, 0), (W, 112), (60, 45, 35), -1)
        cv2.putText(canvas, "Ball Labeler", (40, 66), FONT, 1.6, (255, 255, 255), 3, LINE)
        cv2.putText(canvas, "Help teach the computer to find the tennis ball.",
                    (42, 98), FONT, 0.62, (210, 210, 210), 1, LINE)

        def big_button(y, title, subtitle, color, action):
            x0, y0, x1, y1 = 40, y, W - 40, y + 96
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (235, 235, 235), 2)
            cv2.putText(canvas, title, (x0 + 28, y0 + 44), FONT, 0.95,
                        (255, 255, 255), 2, LINE)
            cv2.putText(canvas, subtitle, (x0 + 28, y0 + 76), FONT, 0.55,
                        (225, 225, 225), 1, LINE)
            state["rects"].append((x0, y0, x1, y1, action))

        big_button(148, "1.  Label a new video",
                   "Mark where the ball is in each picture.",
                   (120, 80, 40), lambda: choose("label"))
        big_button(258, "2.  Fix the computer's guesses",
                   "Check the computer's work and fix any mistakes.",
                   (40, 90, 120), lambda: choose("review"))
        big_button(368, "3.  How do I use this?",
                   "A quick guide to the buttons.",
                   (60, 60, 60), lambda: go("help"))

        x0, y0, x1, y1 = 40, 478, W - 40, 548
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (45, 45, 45), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (150, 150, 150), 1)
        cv2.putText(canvas, "Quit", (x0 + 28, y0 + 46), FONT, 0.8,
                    (210, 180, 180), 2, LINE)
        state["rects"].append((x0, y0, x1, y1, lambda: choose("quit")))

        cv2.putText(canvas, "Tip: you can also press 1, 2, 3, or Q on the keyboard.",
                    (42, H - 22), FONT, 0.5, (150, 150, 150), 1, LINE)
        cv2.imshow(win, canvas)

    def render_help():
        state["rects"] = []
        canvas = np.full((H, W, 3), 28, np.uint8)
        cv2.putText(canvas, "How to use Ball Labeler", (40, 58), FONT, 1.0,
                    (255, 255, 255), 2, LINE)
        y = 104
        for kind, text in WELCOME_HELP:
            if kind == "sp":
                y += 14
                continue
            if kind == "h":
                cv2.putText(canvas, text, (40, y), FONT, 0.6, (150, 200, 255), 1, LINE)
            else:
                cv2.putText(canvas, "  " + text, (40, y), FONT, 0.5,
                            (225, 225, 225), 1, LINE)
            y += 28
        x0, y0, x1, y1 = 40, H - 68, 260, H - 22
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (58, 58, 58), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 220, 150), 1)
        cv2.putText(canvas, "Back", (x0 + 22, y1 - 15), FONT, 0.62,
                    (255, 220, 150), 1, LINE)
        state["rects"].append((x0, y0, x1, y1, lambda: go("home")))
        cv2.imshow(win, canvas)

    while state["result"] is None:
        if state["page"] == "help":
            render_help()
        else:
            render_home()
        key = cv2.waitKey(20) & 0xFF
        if state["page"] == "help":
            if key in (27, ord("q"), ord("b")):
                go("home")
        else:
            if key == ord("1"):
                choose("label")
            elif key == ord("2"):
                choose("review")
            elif key in (ord("3"), ord("h")):
                go("help")
            elif key in (ord("q"), 27):
                choose("quit")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            state["result"] = "quit"
            break

    cv2.destroyWindow(win)
    cv2.waitKey(1)
    return state["result"]


def run_gui(frames_dir, n, width, height, csv_path, progress_path,
            scale, loupe_zoom, ext=IMG_EXT, review=False, fullscreen=False,
            export_path=None):
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for the labelling GUI.")

    if export_path is None:
        export_path = os.path.splitext(csv_path)[0] + "-labeled.csv"

    vis = np.zeros(n, dtype=int)
    xs = np.zeros(n, dtype=int)
    ys = np.zeros(n, dtype=int)
    present = np.zeros(n, dtype=bool)
    if os.path.exists(csv_path):
        vis, xs, ys, present = load_labels(csv_path, n, return_present=True)

    if review:
        # Every frame the model wrote a row for is a pre-filled label to check.
        labelled = present.copy()
        reviewed = load_progress(progress_path, n, key="reviewed")
    else:
        labelled = load_progress(progress_path, n)
        reviewed = None

    screen_w, screen_h = get_screen_size()
    explicit_scale = scale  # honoured for fit/fullscreen; "actual" always uses 1:1

    # Where to start: first thing still needing attention.
    pending = (~reviewed) if review else (~labelled)
    start_idx = int(np.argmax(pending)) if pending.any() else 0

    state = {"idx": start_idx,
             "cursor": None,        # mouse position in original coords
             "loupe_center": None,  # what the magnifier is centred on (stable during nudges)
             "last_idx": -1,        # to detect frame changes
             "dirty": False,
             "fullscreen": bool(fullscreen),
             "display_mode": "fullscreen" if fullscreen else "fit",
             "prev_mode": "fit",    # mode to restore when leaving fullscreen via 'f'
             "scale": 1.0,          # set properly by apply_mode() below
             "disp_w": width, "disp_h": height,
             "x_off": 0,            # where the image sits in the canvas (for mouse mapping)
             "y_off": BAR_H,
             "win_size": None,      # last window size we applied (windowed mode)
             "menu": False,         # a menu screen showing?
             "menu_page": "settings",  # which menu screen: "settings" or "directions"
             "menu_rects": [],      # clickable (x0,y0,x1,y1,action) rows in the menu
             "settings_btn": None,  # clickable Settings button in the status bar
             "flash": "",           # transient confirmation message
             "flash_until": 0.0}

    def fit_scale():
        avail_w = screen_w * 0.98
        avail_h = screen_h * 0.92 - BAR_H
        s = min(avail_w / width, avail_h / height)
        return max(0.1, min(s, 8.0))

    def apply_mode(mode):
        """Switch display mode: 'fit' (scale to screen), 'fullscreen', or
        'actual' (1:1 pixels). Updates scale, image size and the window."""
        state["display_mode"] = mode
        if mode == "actual":
            s, fs = 1.0, False
        elif mode == "fullscreen":
            s, fs = (explicit_scale or fit_scale()), True
        else:  # "fit"
            s, fs = (explicit_scale or fit_scale()), False
        state["scale"] = s
        state["disp_w"], state["disp_h"] = int(width * s), int(height * s)
        state["fullscreen"] = fs
        if fs:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        state["win_size"] = None  # force a resize on the next render

    def flash(msg):
        state["flash"] = msg
        state["flash_until"] = time.time() + 2.5
        print("  " + msg)

    def open_menu():
        state["menu"] = True
        state["win_size"] = None

    def close_menu():
        state["menu"] = False
        state["win_size"] = None

    def do_export():
        save_labels(export_path, vis, xs, ys)
        flash(f"Exported -> {os.path.basename(export_path)}")

    def do_menu_action(action):
        kind, val = action
        if kind == "mode":
            apply_mode(val)
            flash(f"Display: {val}")
        elif kind == "export":
            do_export()
        elif kind == "page":          # switch between settings / directions
            state["menu_page"] = val
            state["win_size"] = None
        elif kind == "back":
            close_menu()

    if review:
        win = "Ball Labeler  -  Check the computer's guesses   (C = keep, click = fix, U = next, Q = save & quit)"
    else:
        win = "Ball Labeler  -  Click the ball   (N = can't see it, A = back, M = menu, Q = save & quit)"

    def to_orig(dx, dy):
        # dx,dy are display coords within the FULL canvas; subtract the image's
        # offset within the canvas (status bar height + any letterbox centring).
        ix = (dx - state["x_off"]) / state["scale"]
        iy = (dy - state["y_off"]) / state["scale"]
        return ix, iy

    def on_mouse(event, x, y, flags, param):
        # Settings screen: clicks pick a menu row.
        if state["menu"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                for (x0, y0, x1, y1, action) in state["menu_rects"]:
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        do_menu_action(action)
                        break
            return
        # Settings button in the status bar opens the menu.
        if event == cv2.EVENT_LBUTTONDOWN and state["settings_btn"]:
            bx0, by0, bx1, by1 = state["settings_btn"]
            if bx0 <= x <= bx1 and by0 <= y <= by1:
                open_menu()
                return
        disp_w, disp_h = state["disp_w"], state["disp_h"]
        xo, yo = state["x_off"], state["y_off"]
        if not (xo <= x < xo + disp_w and yo <= y < yo + disp_h):
            return  # outside the image (status bar or fullscreen letterbox)
        ox, oy = to_orig(x, y)
        state["cursor"] = (ox, oy)
        state["loupe_center"] = (ox, oy)  # moving the mouse re-anchors the magnifier
        if event == cv2.EVENT_LBUTTONDOWN:
            set_label(state["idx"], ox, oy, advance=True)
        elif event == cv2.EVENT_RBUTTONDOWN:
            set_label(state["idx"], ox, oy, advance=False)

    def mark_reviewed(i):
        if reviewed is not None:
            reviewed[i] = True

    def set_label(i, ox, oy, advance):
        xs[i] = int(round(max(0, min(width - 1, ox))))
        ys[i] = int(round(max(0, min(height - 1, oy))))
        vis[i] = 1
        labelled[i] = True
        mark_reviewed(i)  # actively placing a point counts as reviewing it
        state["dirty"] = True
        if advance:
            state["idx"] = min(n - 1, i + 1)

    def set_not_visible(i):
        xs[i] = 0
        ys[i] = 0
        vis[i] = 0
        labelled[i] = True
        mark_reviewed(i)
        state["dirty"] = True
        state["idx"] = min(n - 1, i + 1)

    def confirm(i):
        # Review mode: accept the current (model) label unchanged and advance.
        labelled[i] = True
        mark_reviewed(i)
        state["dirty"] = True
        state["idx"] = min(n - 1, i + 1)

    def clear_label(i):
        xs[i] = 0
        ys[i] = 0
        vis[i] = 0
        labelled[i] = False
        if reviewed is not None:
            reviewed[i] = False
        state["dirty"] = True

    def nudge(i, dx, dy):
        # Micro-adjust the current label by 1 pixel (original-resolution coords).
        if not (labelled[i] and vis[i]):
            return  # nothing to move on an unlabelled / not-visible frame
        xs[i] = int(max(0, min(width - 1, xs[i] + dx)))
        ys[i] = int(max(0, min(height - 1, ys[i] + dy)))
        mark_reviewed(i)  # fine-tuning a point counts as reviewing it
        # Keep the magnifier view fixed; the red marker inside it moves instead.
        # If the label reaches the loupe edge, re-anchor so it stays in view.
        lc = state["loupe_center"]
        if lc is None or abs(xs[i] - lc[0]) > 16 or abs(ys[i] - lc[1]) > 16:
            state["loupe_center"] = (xs[i], ys[i])
        state["dirty"] = True

    def save():
        save_labels(csv_path, vis, xs, ys)
        save_progress(progress_path, labelled, reviewed)
        state["dirty"] = False

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    apply_mode(state["display_mode"])  # sets scale, image size and window state

    frame_cache = {}

    def get_frame(i):
        if i not in frame_cache:
            img = cv2.imread(os.path.join(frames_dir, f"{i}.{ext}"))
            if len(frame_cache) > 8:
                frame_cache.clear()
            frame_cache[i] = img
        return frame_cache[i]

    def render():
        if state["menu"]:
            if state["menu_page"] == "directions":
                render_directions()
            else:
                render_menu()
            return
        scale = state["scale"]
        disp_w, disp_h = state["disp_w"], state["disp_h"]
        i = state["idx"]
        # On arriving at a frame, anchor the magnifier (green crosshair).
        if i != state["last_idx"]:
            state["last_idx"] = i
            if labelled[i] and vis[i]:
                # Already labelled: centre on this frame's own label.
                state["loupe_center"] = (xs[i], ys[i])
            else:
                # Unlabelled: default the green cursor to where the ball was
                # last frame (it usually moves little), so you start right on it.
                prev = i - 1
                while prev >= 0 and not (labelled[prev] and vis[prev]):
                    prev -= 1
                if prev >= 0:
                    state["loupe_center"] = (xs[prev], ys[prev])
                    state["cursor"] = (xs[prev], ys[prev])
        base = get_frame(i)
        if base is None:
            base = np.zeros((height, width, 3), np.uint8)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        disp = cv2.resize(base, (disp_w, disp_h), interpolation=interp)

        # Fading purple trail of the ball's recent path: the most recent
        # position is the brightest (opaque) purple; each earlier position is
        # progressively more translucent.
        TRAIL_LEN = 8
        trail = []                       # most-recent first
        j = i - 1
        while j >= 0 and len(trail) < TRAIL_LEN:
            if labelled[j] and vis[j]:
                trail.append((xs[j], ys[j]))
            j -= 1
        # Draw oldest (faintest) first so the brightest recent dot sits on top.
        for k in range(len(trail) - 1, -1, -1):
            bx, by = trail[k]            # k == 0 is the most recent position
            alpha = max(0.12, 1.0 - 0.85 * (k / max(1, TRAIL_LEN - 1)))
            _blend_circle(disp, (bx * scale, by * scale), 5,
                          (240, 32, 160), alpha)  # purple (BGR)

        # Yellow = a model prediction not yet reviewed; red = reviewed/edited.
        unconfirmed = bool(review and reviewed is not None and not reviewed[i])
        mark_color = (0, 255, 255) if unconfirmed else (0, 0, 255)

        # Current frame's label marker.
        if labelled[i] and vis[i]:
            px, py = int(xs[i] * scale), int(ys[i] * scale)
            cv2.circle(disp, (px, py), 7, mark_color, 2)
            cv2.drawMarker(disp, (px, py), mark_color, cv2.MARKER_CROSS, 16, 1)

        # Loupe magnifier (top-right). The VIEW stays put while you nudge; a red
        # marker inside it shows the label pixel and moves as you press arrows.
        anchor = state["loupe_center"] if state["loupe_center"] is not None else state["cursor"]
        if anchor is not None:
            ox, oy = anchor
            r = 22
            x0, y0 = int(round(ox - r)), int(round(oy - r))
            crop = np.zeros((2 * r, 2 * r, 3), np.uint8)
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(width, x0 + 2 * r), min(height, y0 + 2 * r)
            if sx1 > sx0 and sy1 > sy0:
                patch = base[sy0:sy1, sx0:sx1]
                crop[sy0 - y0:sy0 - y0 + patch.shape[0],
                     sx0 - x0:sx0 - x0 + patch.shape[1]] = patch
            loupe = cv2.resize(crop, (r * 2 * loupe_zoom, r * 2 * loupe_zoom),
                               interpolation=cv2.INTER_NEAREST)
            lh, lw = loupe.shape[:2]
            # Stationary green reference crosshair at the magnifier centre.
            cv2.line(loupe, (lw // 2, 0), (lw // 2, lh), (0, 180, 0), 1)
            cv2.line(loupe, (0, lh // 2), (lw, lh // 2), (0, 180, 0), 1)
            # Red marker at the actual label pixel - moves 1px per arrow press.
            if labelled[i] and vis[i]:
                lx = int((xs[i] - x0) * loupe_zoom + loupe_zoom // 2)
                ly = int((ys[i] - y0) * loupe_zoom + loupe_zoom // 2)
                if 0 <= lx < lw and 0 <= ly < lh:
                    cv2.drawMarker(loupe, (lx, ly), mark_color, cv2.MARKER_CROSS, 14, 1)
                    cv2.circle(loupe, (lx, ly), 3, mark_color, -1)
            disp[0:lh, disp_w - lw:disp_w] = loupe

        # Compose the canvas. In fullscreen the canvas is the whole screen and
        # the image is centred (letterboxed); otherwise it's a snug bar+image.
        if state["fullscreen"]:
            cw, ch = screen_w, screen_h
        else:
            cw, ch = disp_w, disp_h + BAR_H
        canvas = np.zeros((ch, cw, 3), np.uint8)
        x_off = max(0, (cw - disp_w) // 2)
        y_off = max(BAR_H, BAR_H + (ch - BAR_H - disp_h) // 2)
        state["x_off"], state["y_off"] = x_off, y_off
        ph = min(disp_h, ch - y_off)
        pw = min(disp_w, cw - x_off)
        canvas[y_off:y_off + ph, x_off:x_off + pw] = disp[:ph, :pw]
        # Keep the windowed window sized to the canvas (1:1, so clicks map exactly).
        if not state["fullscreen"] and state["win_size"] != (cw, ch):
            cv2.resizeWindow(win, cw, ch)
            state["win_size"] = (cw, ch)
        friendly_state = ('Ball: visible' if (labelled[i] and vis[i])
                          else ('Ball: not visible' if labelled[i]
                                else 'Ball: not marked yet'))
        if review:
            done = int(reviewed.sum())
            bar1 = f"Checking picture {i+1} of {n}     Done {done}/{n}     {friendly_state}"
            bar2 = ("C = guess looks right (keep it)    Click = fix it    "
                    "N = can't see ball    U = next to check    M = help/menu    Q = save & quit")
        else:
            done = int(labelled.sum())
            bar1 = f"Picture {i+1} of {n}     Done {done}/{n}     {friendly_state}"
            bar2 = ("Click the ball = mark it & go next    N = can't see ball    "
                    "A = back    M = help/menu    Q = save & quit")
        cv2.putText(canvas, bar1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, bar2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 220, 180), 1, cv2.LINE_AA)
        # Settings button (top-right of the status bar) - opens the menu.
        bw, bh = 120, 26
        bx1, by0 = cw - 10, 6
        bx0, by1 = bx1 - bw, by0 + bh
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (70, 70, 70), -1)
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (170, 170, 170), 1)
        cv2.putText(canvas, "Settings (m)", (bx0 + 9, by1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
        state["settings_btn"] = (bx0, by0, bx1, by1)
        if state["dirty"]:
            cv2.circle(canvas, (bx0 - 16, 18), 6, (0, 0, 255), -1)
        cv2.imshow(win, canvas)

    def render_menu():
        if state["fullscreen"]:
            mw, mh = screen_w, screen_h
        else:
            mw = max(state["disp_w"], 720)
            mh = max(state["disp_h"] + BAR_H, 540)
            if state["win_size"] != (mw, mh):
                cv2.resizeWindow(win, mw, mh)
                state["win_size"] = (mw, mh)
        FONT, LINE = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        canvas = np.full((mh, mw, 3), 32, np.uint8)
        cv2.putText(canvas, "SETTINGS", (40, 64), FONT, 1.1, (255, 255, 255), 2, LINE)
        cv2.putText(canvas, "Click an option below.   (Esc or m = back to labeling)",
                    (42, 98), FONT, 0.52, (170, 170, 170), 1, LINE)
        rects = []
        y = [142]

        def section(title):
            cv2.putText(canvas, title, (40, y[0] + 16), FONT, 0.6, (150, 200, 255), 1, LINE)
            y[0] += 34

        def row(label, active, action, color=(235, 235, 235)):
            x0, y0, x1, y1 = 40, y[0], mw - 40, y[0] + 50
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (58, 58, 58), -1)
            if active:
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 190, 0), 2)
            cv2.putText(canvas, ("[x] " if active else "[ ] ") + label,
                        (x0 + 18, y0 + 33), FONT, 0.62, color, 1, LINE)
            rects.append((x0, y0, x1, y1, action))
            y[0] += 62

        section("Display mode")
        row("1   Fit to screen", state["display_mode"] == "fit", ("mode", "fit"))
        row("2   Fullscreen", state["display_mode"] == "fullscreen", ("mode", "fullscreen"))
        row("3   Actual size (1:1 pixels)", state["display_mode"] == "actual", ("mode", "actual"))
        y[0] += 14
        section("Export")
        row("E   Export labeled CSV  ->  " + os.path.basename(export_path),
            False, ("export", None), (185, 255, 185))
        y[0] += 14
        section("Help")
        row("D   Directions - how to use the software",
            False, ("page", "directions"), (200, 220, 255))
        y[0] += 14
        row("Back to labeling", False, ("back", None), (255, 220, 150))
        state["menu_rects"] = rects
        if state["flash"] and time.time() < state["flash_until"]:
            cv2.putText(canvas, state["flash"], (40, mh - 28), FONT, 0.62,
                        (130, 255, 130), 1, LINE)
        cv2.imshow(win, canvas)

    # Lines of the Directions screen. ("h"=heading, "t"=text, "sp"=spacer)
    DIRECTIONS = [
        ("h", "What this tool does"),
        ("t", "Step through a clip frame by frame and mark where the ball is, so"),
        ("t", "the model has clean training labels. In review mode you check the"),
        ("t", "model's own guesses and fix the wrong ones."),
        ("sp", ""),
        ("h", "Moving between frames"),
        ("t", "d / a = next / previous frame      w / e = jump back / forward 10"),
        ("t", "u = jump to the next frame that still needs you"),
        ("sp", ""),
        ("h", "Marking the ball"),
        ("t", "Left-click the ball = set its position and go to the next frame"),
        ("t", "Right-click or '/'  = set position WITHOUT advancing (fine-tune)"),
        ("t", "Arrow keys = nudge the mark one pixel at a time"),
        ("t", "n or Space = ball not visible (occluded or off-screen)"),
        ("t", "z = clear this frame's mark"),
        ("sp", ""),
        ("h", "Reviewing model predictions"),
        ("t", "c or Enter = the prediction is correct, confirm it and advance"),
        ("t", "Yellow mark = not reviewed yet     Red mark = reviewed or edited"),
        ("t", "Purple dots = the ball's recent path (brightest = most recent)"),
        ("t", "The top-right magnifier shows a zoomed view; green lines = centre"),
        ("sp", ""),
        ("h", "Display, export and saving"),
        ("t", "f = fullscreen     m = settings (display mode, export, this page)"),
        ("t", "Export writes every frame to <video name>-labeled.csv next to the video"),
        ("t", "s = save now     q or Esc = save and quit (it also auto-saves on quit)"),
    ]

    def render_directions():
        needed = 150 + len(DIRECTIONS) * 30 + 90
        if state["fullscreen"]:
            mw, mh = screen_w, screen_h
        else:
            mw = max(state["disp_w"], 780)
            mh = max(state["disp_h"] + BAR_H, needed, 560)
            if state["win_size"] != (mw, mh):
                cv2.resizeWindow(win, mw, mh)
                state["win_size"] = (mw, mh)
        FONT, LINE = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        canvas = np.full((mh, mw, 3), 28, np.uint8)
        cv2.putText(canvas, "DIRECTIONS", (40, 64), FONT, 1.1, (255, 255, 255), 2, LINE)
        cv2.putText(canvas, "How to use the labeler.   (Esc or m = back to settings)",
                    (42, 98), FONT, 0.52, (170, 170, 170), 1, LINE)
        y = 138
        avail = mh - y - 90
        line_h = max(22, min(32, avail // max(1, len(DIRECTIONS))))
        for kind, text in DIRECTIONS:
            if kind == "sp":
                y += line_h // 2
                continue
            if kind == "h":
                cv2.putText(canvas, text, (40, y + int(line_h * 0.72)),
                            FONT, 0.6, (150, 200, 255), 1, LINE)
            else:
                cv2.putText(canvas, "  " + text, (40, y + int(line_h * 0.72)),
                            FONT, 0.48, (225, 225, 225), 1, LINE)
            y += line_h
        # Back-to-settings button.
        bw, bh = 220, 44
        x0, y0 = 40, mh - bh - 16
        x1, y1 = x0 + bw, y0 + bh
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (58, 58, 58), -1)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 220, 150), 1)
        cv2.putText(canvas, "Back to settings", (x0 + 16, y1 - 15),
                    FONT, 0.58, (255, 220, 150), 1, LINE)
        state["menu_rects"] = [(x0, y0, x1, y1, ("page", "settings"))]
        cv2.imshow(win, canvas)

    if review:
        nrev = int(reviewed.sum())
        print(f"\nReviewing model CSV: {csv_path}")
        print(f"  {n} frames, {nrev} already reviewed. Corrections overwrite this CSV.")
        print("  c/Enter = confirm correct, click/nudge/n = fix, u = next unreviewed.\n")
    else:
        print(f"\nLabelling {n} frames. Output -> {csv_path}")
        print("Window controls are shown on screen. Close with q or Esc.\n")

    while True:
        render()
        keyx = cv2.waitKeyEx(20)  # full key code (waitKey & 0xFF drops arrows on Windows)
        key = keyx & 0xFF

        # Menu screens have their own keys; handle and skip labelling controls.
        if state["menu"]:
            if state["menu_page"] == "directions":
                if key in (27, ord("q"), ord("m"), ord("b")):
                    do_menu_action(("page", "settings"))
            else:  # settings page
                if key in (27, ord("q"), ord("m")):
                    close_menu()
                elif key == ord("1"):
                    apply_mode("fit"); flash("Display: fit to screen")
                elif key == ord("2"):
                    apply_mode("fullscreen"); flash("Display: fullscreen")
                elif key == ord("3"):
                    apply_mode("actual"); flash("Display: actual size (1:1)")
                elif key == ord("d"):
                    do_menu_action(("page", "directions"))
                elif key in (ord("e"), 13, 10):
                    do_export()
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        i = state["idx"]
        if keyx in ARROW_LEFT:            # nudge current label 1 px
            nudge(i, -1, 0)
        elif keyx in ARROW_RIGHT:
            nudge(i, 1, 0)
        elif keyx in ARROW_UP:
            nudge(i, 0, -1)
        elif keyx in ARROW_DOWN:
            nudge(i, 0, 1)
        elif key == ord("/"):             # set at cursor (same as right-click)
            if state["cursor"] is not None:
                cx, cy = state["cursor"]
                set_label(i, cx, cy, advance=False)
        elif review and (key == ord("c") or keyx in (13, 10)):  # confirm as-is
            confirm(i)
        elif key in (ord("q"), 27):  # q / Esc
            break
        elif key == ord("d"):             # next
            state["idx"] = min(n - 1, i + 1)
        elif key == ord("a"):             # prev
            state["idx"] = max(0, i - 1)
        elif key == ord("e"):             # +10
            state["idx"] = min(n - 1, i + 10)
        elif key == ord("w"):             # -10
            state["idx"] = max(0, i - 10)
        elif key in (ord("n"), ord(" ")):
            set_not_visible(i)
        elif key == ord("z"):
            clear_label(i)
        elif key == ord("u"):
            todo = (~reviewed) if review else (~labelled)
            if todo.any():
                state["idx"] = int(np.argmax(todo))
        elif key == ord("f"):             # quick toggle fullscreen <-> previous mode
            if state["display_mode"] == "fullscreen":
                apply_mode(state["prev_mode"])
            else:
                state["prev_mode"] = state["display_mode"]
                apply_mode("fullscreen")
        elif key == ord("m"):             # open settings menu
            open_menu()
        elif key == ord("s"):
            save()
            print("  saved.")
        # window closed via the X button
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    save()
    cv2.destroyAllWindows()
    if review:
        print(f"\nDone. {int(reviewed.sum())}/{n} frames reviewed. CSV: {csv_path}")
    else:
        print(f"\nDone. {int(labelled.sum())}/{n} frames labelled. CSV: {csv_path}")
    return csv_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def run_one_session(video, review, review_csv, args):
    """Prepare frames for one clip and open the labeling window.

    Shared by the command-line path and the click-only home screen. Prints
    friendly messages and returns quietly on problems instead of exiting, so the
    home screen can loop back to its menu.
    """
    if not os.path.exists(video):
        print(f"  Couldn't find that video: {video}")
        return

    stem = os.path.splitext(os.path.basename(video))[0]
    if review:
        out_csv = os.path.abspath(review_csv)  # overwrite the model CSV in place
    else:
        out_csv = args.out or os.path.join(os.path.dirname(os.path.abspath(video)),
                                           f"{stem}_ball.csv")
    frames_dir = args.frames_dir or os.path.join(
        os.path.dirname(os.path.abspath(video)), f"{stem}_frames")
    progress_path = out_csv + ".progress.json"
    export_path = os.path.join(os.path.dirname(os.path.abspath(video)),
                               f"{stem}-labeled.csv")

    print(f"Getting the video ready: {video}")
    n, width, height = extract_frames(video, frames_dir)
    if n == 0:
        print("  Couldn't read any pictures from that video. Try a different file "
              "(an .mp4 usually works best).")
        return
    print(f"  {n} pictures, {width}x{height}")

    if review:
        _, _, _, present = load_labels(out_csv, n, return_present=True)
        rows = int(present.sum())
        print(f"  The computer made a guess on {rows} of {n} pictures.")
        if rows < n:
            print("  Note: some pictures have no guess yet and will show as "
                  "'not marked yet'. Make sure the video and the guesses go together.")

    run_gui(frames_dir, n, width, height, out_csv, progress_path,
            args.scale, args.loupe_zoom, review=review, fullscreen=args.fullscreen,
            export_path=export_path)

    if args.dataset_root:
        rally = args.rally or stem
        dv, dc = place_in_dataset(video, out_csv, args.dataset_root,
                                  args.split, args.match, rally)
        print(f"Placed in dataset:\n  {dv}\n  {dc}")


def welcome_loop(args):
    """No-typing home screen: pick an action, do it, then come back here."""
    while True:
        choice = run_welcome()
        if choice == "quit":
            print("All done. Bye!")
            return
        if choice == "label":
            video = args.video or pick_video_dialog()
            if not video:
                continue  # cancelled the file picker -> back to the menu
            run_one_session(video, False, None, args)
        elif choice == "review":
            review_csv = args.csv or pick_csv_dialog()
            if not review_csv or not os.path.exists(review_csv):
                continue
            video = args.video or pick_video_dialog()
            if not video:
                continue
            run_one_session(video, True, review_csv, args)


def main(argv=None):
    p = argparse.ArgumentParser(description="Click-to-label ball positions for TrackNetV3.")
    p.add_argument("--video", default=None, help="Path to the (trimmed) rally clip.")
    p.add_argument("--out", default=None, help="Output CSV (default: <video>_ball.csv).")
    p.add_argument("--review", action="store_true",
                   help="Review/verify a model-produced CSV frame by frame. A file "
                        "picker lets you choose the CSV (and the video if --video is "
                        "omitted); corrections are saved back over the same CSV.")
    p.add_argument("--csv", default=None,
                   help="(review) Model CSV to verify. If omitted in --review mode, a "
                        "file-picker dialog opens so you can choose it.")
    p.add_argument("--frames-dir", default=None,
                   help="Where to cache extracted frames (default: <video>_frames/).")
    p.add_argument("--scale", type=float, default=None,
                   help="Display scale (default: auto-fit to ~1280x720).")
    p.add_argument("--loupe-zoom", type=int, default=5, help="Magnifier zoom factor.")
    p.add_argument("--fullscreen", action="store_true",
                   help="Start in borderless fullscreen (toggle anytime with the 'f' key).")
    p.add_argument("--dataset-root", default=None,
                   help="If set, copy clip+CSV into a TrackNetV3 dataset layout here.")
    p.add_argument("--split", default="train", choices=["train", "test", "val"])
    p.add_argument("--match", default="24", help="Match number for dataset placement.")
    p.add_argument("--rally", default=None,
                   help="Rally name for dataset placement (default: video stem).")
    args = p.parse_args(argv)

    # No options given (e.g. double-clicked the launcher): open the easy,
    # click-only home screen and loop there until the user quits.
    if not args.video and not args.review and not args.csv:
        welcome_loop(args)
        return

    review = args.review
    review_csv = None

    if review:
        # 1) The model CSV to verify (overwritten in place when you save).
        review_csv = args.csv or pick_csv_dialog()
        if not review_csv:
            p.error("No CSV selected to review.")
        if not os.path.exists(review_csv):
            p.error(f"CSV not found: {review_csv}")
        # 2) The matching video, so we can show frames behind the predictions.
        video = args.video or pick_video_dialog()
        if not video:
            p.error("No video selected - a video is needed to show the frames.")
    else:
        video = args.video

    if not video:
        p.error("--video is required (or run with no options for the easy menu).")
    if not os.path.exists(video):
        p.error(f"Video not found: {video}")

    run_one_session(video, review, review_csv, args)


if __name__ == "__main__":
    main()
