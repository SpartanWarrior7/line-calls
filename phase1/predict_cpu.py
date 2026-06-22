"""
CPU-compatible TrackNetV3 + InpaintNet inference, chunked for the 45s bash limit.

Mirrors the official predict.py pipeline exactly on CPU.
Run repeatedly until "All done":
    python predict_cpu.py            # auto-advances through stages
    python predict_cpu.py --finish   # force InpaintNet stage (after all TrackNet windows done)
    python predict_cpu.py --reset    # wipe checkpoints and start over

Output: ckpt/detections.json (compatible with run_chunked stage_finish)
        ckpt/ball.csv        (Frame, Visibility, X, Y — official format)
"""

import argparse, json, math, sys, time
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
import pandas as pd

# ---- Repo path setup -------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent / "TrackNetV3"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import TrackNet, InpaintNet                          # noqa: E402
from utils.general import get_model, WIDTH, HEIGHT, COOR_TH     # noqa: E402
from dataset import Shuttlecock_Trajectory_Dataset              # noqa: E402

TRACKNET_CKPT   = str(REPO / "ckpts" / "TrackNet_best.pt")
INPAINTNET_CKPT = str(REPO / "ckpts" / "InpaintNet_best.pt")
CKPT_DIR  = Path("ckpt")
CHUNK_SIZE = 12      # windows per call  (12 × ~3s ≈ 36s, safely under 45s)
DEVICE     = "cpu"


# ============================================================================
# Helpers — inlined from test.py to avoid pycocotools dependency
# ============================================================================

def _predict_location(heatmap):
    """Return (x, y, w, h) bounding box of largest blob in a binary uint8 heatmap."""
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0
    cnts, _ = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts]
    best  = max(range(len(rects)), key=lambda i: rects[i][2] * rects[i][3])
    return rects[best]


def _generate_inpaint_mask(pred_dict, th_h=30, max_gap=15):
    """
    Official generate_inpaint_mask logic with a max_gap guard.

    max_gap: maximum number of consecutive invisible frames to inpaint.
    Gaps larger than this are between-rally breaks (ball off-screen) and
    must NOT be inpainted — InpaintNet would hallucinate a smooth path
    between two unrelated ball positions.
    """
    y        = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    mask     = np.zeros(len(y), dtype=np.int32)
    i = j = 0
    while j < len(vis_pred):
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        gap = j - i
        if gap <= max_gap:          # only fill short within-rally gaps
            if i == 0 and y[j] > th_h:
                mask[:j] = 1
            elif (i > 1 and y[i - 1] > th_h) and (j < len(vis_pred) and y[j] > th_h):
                mask[i:j] = 1
        i = j + 1
    return mask.tolist()


# ============================================================================
# Model loading
# ============================================================================

def _load_tracknet():
    ckpt    = torch.load(TRACKNET_CKPT, map_location=DEVICE)
    seq_len = ckpt['param_dict']['seq_len']
    bg_mode = ckpt['param_dict']['bg_mode']
    model   = get_model('TrackNet', seq_len, bg_mode).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  TrackNet loaded (seq_len={seq_len}, bg_mode={bg_mode})")
    return model, seq_len, bg_mode


def _load_inpaintnet():
    ckpt    = torch.load(INPAINTNET_CKPT, map_location=DEVICE)
    seq_len = ckpt['param_dict']['seq_len']
    model   = get_model('InpaintNet').to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  InpaintNet loaded (seq_len={seq_len})")
    return model, seq_len


# ============================================================================
# Preprocessing — mirrors Shuttlecock_Trajectory_Dataset.__getitem__ exactly
# (bg_mode='concat', uint8 PIL resize, /255 at end)
# ============================================================================

def _compute_median(video_path, max_frames=30):
    """
    Compute per-pixel median over up to max_frames evenly-spaced frames.

    Reads the video SEQUENTIALLY (no seeking — sequential decode is ~10×
    faster than repeated cap.set() on H.264).  Each sampled frame is resized
    to model input resolution (HEIGHT×WIDTH) before storage, so the working
    set is only max_frames × 288 × 512 × 3 ≈ 13 MB instead of ~1.5 GB.

    Returns (HEIGHT, WIDTH, 3) uint8 RGB — already at model resolution.
    """
    cap  = cv2.VideoCapture(str(video_path))
    n    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, n // max_frames)
    frames_small = []
    fi = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if fi % step == 0:
            # Resize while still BGR (channel order irrelevant for median), then flip
            small = cv2.resize(f, (WIDTH, HEIGHT))[:, :, ::-1]  # (H,W,3) RGB uint8
            frames_small.append(small.copy())
        fi += 1
    cap.release()
    return np.median(frames_small, axis=0).astype(np.uint8)  # (HEIGHT, WIDTH, 3)


def _median_to_model_input(median_small):
    """
    Convert (HEIGHT, WIDTH, 3) uint8 RGB median to (3, HEIGHT, WIDTH) float32.
    The median is already at model resolution — just channel-first + /255.
    """
    return np.moveaxis(median_small, -1, 0).astype(np.float32) / 255.  # (3, H, W)


def _preprocess_seq(median_model, frames_bgr):
    """
    Build a (27, HEIGHT, WIDTH) float32 array for one 8-frame window.
    Matches the dataset's bg_mode='concat' __getitem__:
      - PIL-equivalent resize via cv2.resize (INTER_LINEAR, same result)
      - channel order: bg(3ch) | frame_0(3ch) | … | frame_7(3ch)
      - normalised /255 at end

      median_model: (3, H, W) float32 in [0,1]  — background
      frames_bgr:   list of seq_len uint8 BGR frames at original resolution
    """
    channels = [median_model]   # background first
    for frame in frames_bgr:
        # Resize BGR, then flip to RGB, then channel-first
        small = cv2.resize(frame, (WIDTH, HEIGHT))         # (H, W, 3) BGR
        rgb   = small[:, :, ::-1].astype(np.float32)      # (H, W, 3) RGB
        channels.append(np.moveaxis(rgb, -1, 0) / 255.)   # (3, H, W) /255
    return np.concatenate(channels, axis=0)  # (27, H, W)


# ============================================================================
# Stage 1 — background median + meta
# ============================================================================

def stage_setup(video_path):
    CKPT_DIR.mkdir(exist_ok=True)
    print("Computing background median …")

    cap    = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    median_small = _compute_median(str(video_path))
    np.save(str(CKPT_DIR / "bg_median.npy"), median_small)

    # These values are fixed for the TrackNet_best.pt checkpoint
    # (confirmed by param_dict inspection — avoids loading 130 MB of weights here)
    seq_len = 8
    bg_mode = 'concat'

    n_windows = math.ceil(n / seq_len)
    meta = {
        "video": str(Path(video_path).resolve()),
        "fps": fps, "orig_w": orig_w, "orig_h": orig_h,
        "n_frames": n, "n_windows": n_windows,
        "seq_len": seq_len, "bg_mode": bg_mode,
        "w_scaler": orig_w / WIDTH, "h_scaler": orig_h / HEIGHT,
        "next_window": 0,
    }
    (CKPT_DIR / "predict_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  {n} frames @ {fps:.1f}fps  {orig_w}×{orig_h}")
    print(f"  {n_windows} windows to process.")


# ============================================================================
# Stage 2 — TrackNet inference chunk
# ============================================================================

def stage_infer(video_path):
    meta_path = CKPT_DIR / "predict_meta.json"
    if not meta_path.exists():
        stage_setup(video_path)

    meta      = json.loads(meta_path.read_text())
    next_w    = meta["next_window"]
    n_windows = meta["n_windows"]
    n_frames  = meta["n_frames"]
    seq_len   = meta["seq_len"]
    fps       = meta["fps"]
    orig_w    = meta["orig_w"]
    orig_h    = meta["orig_h"]
    w_scaler  = meta["w_scaler"]
    h_scaler  = meta["h_scaler"]

    if next_w >= n_windows:
        print("All windows done — run --finish")
        return

    img_scaler = (w_scaler, h_scaler)
    end_w = min(next_w + CHUNK_SIZE, n_windows)
    print(f"TrackNet: windows {next_w}–{end_w-1} of {n_windows}")

    # Load background and model
    median_rgb   = np.load(str(CKPT_DIR / "bg_median.npy"))
    median_model = _median_to_model_input(median_rgb)    # (3, H, W) /255
    model, _, _  = _load_tracknet()

    # Load existing detections
    raw_path = CKPT_DIR / "raw_tracknet.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}

    # Open video and seek to start of this chunk
    cap = cv2.VideoCapture(str(video_path))
    start_frame = next_w * seq_len
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    t0 = time.perf_counter()

    for w in range(next_w, end_w):
        # Read exactly seq_len frames
        seq = []
        for _ in range(seq_len):
            ok, frame = cap.read()
            if not ok:
                break
            seq.append(frame)

        if not seq:
            break

        # Pad last window if needed
        while len(seq) < seq_len:
            seq.append(seq[-1])

        # Preprocess → (1, 27, H, W)
        x_np = _preprocess_seq(median_model, seq)
        x_t  = torch.from_numpy(x_np).unsqueeze(0).float()

        with torch.no_grad():
            y_pred = model(x_t).squeeze(0).cpu().numpy()   # (seq_len, H, W)

        for f in range(seq_len):
            f_i = w * seq_len + f
            if f_i >= n_frames:
                break

            hm_bin = (y_pred[f] > 0.5).astype(np.uint8)
            bx, by, bw, bh = _predict_location(hm_bin)
            cx = int((bx + bw / 2) * img_scaler[0])
            cy = int((by + bh / 2) * img_scaler[1])
            vis = 0 if cx == 0 and cy == 0 else 1
            raw[str(f_i)] = {"X": cx, "Y": cy, "Visibility": vis}

    cap.release()
    elapsed = time.perf_counter() - t0

    raw_path.write_text(json.dumps(raw))
    meta["next_window"] = end_w
    meta_path.write_text(json.dumps(meta, indent=2))

    n_vis = sum(1 for v in raw.values() if v["Visibility"])
    print(f"  {n_vis}/{len(raw)} visible  |  {100*end_w/n_windows:.0f}%  |  {elapsed:.1f}s")
    if end_w < n_windows:
        rem = (n_windows - end_w) * elapsed / (end_w - next_w)
        print(f"  ~{rem:.0f}s remaining — run again")
    else:
        print("  Inference complete — run --finish")


# ============================================================================
# Stage 3 — InpaintNet + write outputs
# ============================================================================

def stage_finish():
    meta_path = CKPT_DIR / "predict_meta.json"
    raw_path  = CKPT_DIR / "raw_tracknet.json"
    if not raw_path.exists():
        print("No raw_tracknet.json — run inference first."); sys.exit(1)

    meta     = json.loads(meta_path.read_text())
    raw      = json.loads(raw_path.read_text())
    n_frames = meta["n_frames"]
    orig_w   = meta["orig_w"]
    orig_h   = meta["orig_h"]
    fps      = meta["fps"]
    w_scaler = meta["w_scaler"]
    h_scaler = meta["h_scaler"]

    # ---- Build per-frame TrackNet prediction dict (official format) --------
    print("Building TrackNet pred_dict …")
    tracknet_pred = {
        'Frame':      list(range(n_frames)),
        'X':          [raw.get(str(i), {"X": 0})["X"]          for i in range(n_frames)],
        'Y':          [raw.get(str(i), {"Y": 0})["Y"]          for i in range(n_frames)],
        'Visibility': [raw.get(str(i), {"Visibility": 0})["Visibility"] for i in range(n_frames)],
        'Img_scaler': (w_scaler, h_scaler),
        'Img_shape':  (orig_w, orig_h),
    }
    raw_vis_rate = sum(tracknet_pred['Visibility']) / n_frames
    print(f"  Raw detection rate: {raw_vis_rate*100:.1f}%  ({sum(tracknet_pred['Visibility'])}/{n_frames})")

    # ---- Generate inpaint mask ---------------------------------------------
    # max_gap=15: only fill short within-rally occlusions (≤0.5s at 30fps).
    # Longer gaps are between-rally breaks; InpaintNet would hallucinate there.
    MAX_INPAINT_GAP = 15
    tracknet_pred['Inpaint_Mask'] = _generate_inpaint_mask(
        tracknet_pred, th_h=orig_h * 0.05, max_gap=MAX_INPAINT_GAP
    )
    n_to_inpaint = sum(tracknet_pred['Inpaint_Mask'])
    print(f"  Frames flagged for inpainting: {n_to_inpaint}")

    if n_to_inpaint == 0:
        print("  No gaps to fill — using TrackNet output directly.")
        final_X   = tracknet_pred['X']
        final_Y   = tracknet_pred['Y']
        final_Vis = tracknet_pred['Visibility']
    else:
        # ---- Build InpaintNet dataset + run --------------------------------
        inpaintnet, inpaint_seq_len = _load_inpaintnet()
        print(f"  Building InpaintNet dataset (seq_len={inpaint_seq_len}) …")

        dataset = Shuttlecock_Trajectory_Dataset(
            seq_len=inpaint_seq_len,
            sliding_step=inpaint_seq_len,
            data_mode='coordinate',
            pred_dict=tracknet_pred,
            padding=True,
        )
        n_inpaint_win = len(dataset)
        print(f"  Running InpaintNet on {n_inpaint_win} windows …")

        # Accumulate blended outputs (one entry per frame)
        out_X   = list(tracknet_pred['X'])      # start from TrackNet
        out_Y   = list(tracknet_pred['Y'])
        out_Vis = list(tracknet_pred['Visibility'])

        t0 = time.perf_counter()
        for idx in range(n_inpaint_win):
            data_idx, coor_pred, inpaint_mask = dataset[idx]
            # coor_pred:    (L, 2)  normalized by orig_w, orig_h
            # inpaint_mask: (L, 1)  1=invisible/needs inpainting
            c_t = torch.from_numpy(coor_pred).unsqueeze(0).float()    # (1, L, 2)
            m_t = torch.from_numpy(inpaint_mask).unsqueeze(0).float()  # (1, L, 1)

            with torch.no_grad():
                c_out = inpaintnet(c_t, m_t)  # (1, L, 2)

            # Official blend: inpainted where masked, original where detected
            c_blend = c_out * m_t + c_t * (1 - m_t)
            c_blend = c_blend.squeeze(0).cpu().numpy()  # (L, 2)

            # COOR_TH zero-out (suppress near-zero predictions)
            near_zero = (c_blend[:, 0] < COOR_TH) & (c_blend[:, 1] < COOR_TH)
            c_blend[near_zero] = 0.0

            for f in range(inpaint_seq_len):
                f_i = int(data_idx[f][1])
                if f_i >= n_frames:
                    continue
                # Only update frames that InpaintNet was asked to fill
                if inpaint_mask[f, 0] > 0.5:
                    # Convert normalized → original pixel coords
                    # Official: cx = norm_x * WIDTH * w_scaler = norm_x * orig_w
                    cx = int(c_blend[f, 0] * WIDTH * w_scaler)
                    cy = int(c_blend[f, 1] * HEIGHT * h_scaler)
                    vis = 0 if cx == 0 and cy == 0 else 1
                    out_X[f_i]   = cx
                    out_Y[f_i]   = cy
                    out_Vis[f_i] = vis

        elapsed = time.perf_counter() - t0
        print(f"  InpaintNet done in {elapsed:.1f}s")

        final_X, final_Y, final_Vis = out_X, out_Y, out_Vis

    after_vis = sum(final_Vis) / n_frames
    print(f"Final detection rate: {after_vis*100:.1f}%  ({sum(final_Vis)}/{n_frames})")

    # ---- Write ball.csv (official format) ----------------------------------
    csv_path = CKPT_DIR / "ball.csv"
    pd.DataFrame({
        "Frame":      list(range(n_frames)),
        "Visibility": final_Vis,
        "X":          final_X,
        "Y":          final_Y,
    }).to_csv(str(csv_path), index=False)
    print(f"Saved → {csv_path}")

    # ---- Write detections.json (run_chunked compatible) --------------------
    # confidence=1.0  → directly detected by TrackNet
    # confidence=0.5  → gap-filled by InpaintNet (within-rally only)
    raw_conf = {int(k): v["Visibility"] for k, v in
                json.loads((CKPT_DIR / "raw_tracknet.json").read_text()).items()}
    det = {}
    for fi in range(n_frames):
        if final_Vis[fi]:
            conf = 1.0 if raw_conf.get(fi, 0) == 1 else 0.5
            det[str(fi)] = {
                "frame": fi, "t": fi / fps,
                "x": float(final_X[fi]), "y": float(final_Y[fi]),
                "confidence": conf,
            }
    (CKPT_DIR / "detections.json").write_text(json.dumps(det))
    n_real   = sum(1 for d in det.values() if d["confidence"] == 1.0)
    n_filled = sum(1 for d in det.values() if d["confidence"] == 0.5)
    print(f"Wrote detections.json  ({n_real} TrackNet + {n_filled} InpaintNet-filled)")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  default="line-training-6.mp4")
    parser.add_argument("--reset",  action="store_true",
                        help="Delete checkpoints and start over")
    parser.add_argument("--finish", action="store_true",
                        help="Skip to InpaintNet stage (assumes all TrackNet windows done)")
    args = parser.parse_args()


    if args.reset:
        for name in ["predict_meta.json", "raw_tracknet.json",
                     "bg_median.npy", "ball.csv", "detections.json"]:
            p = CKPT_DIR / name
            if p.exists():
                try:
                    p.unlink()
                except PermissionError:
                    p.write_bytes(b"")
                print(f"  Cleared {p}")
        print("Reset done.")
        import sys as _s; _s.exit(0)

    if args.finish:
        stage_finish()
        import sys as _s; _s.exit(0)

    # Auto: advance to the next pending stage
    meta_path = CKPT_DIR / "predict_meta.json"
    if not meta_path.exists() or not (CKPT_DIR / "bg_median.npy").exists():
        stage_setup(args.video)
    else:
        meta = json.loads(meta_path.read_text())
        if meta["next_window"] < meta["n_windows"]:
            stage_infer(args.video)
        elif not (CKPT_DIR / "ball.csv").exists():
            stage_finish()
        else:
            print("All done. ckpt/ball.csv and ckpt/detections.json are ready.")
