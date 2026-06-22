"""
TrackNetV3 ball detector wrapper.

Assumes you have cloned https://github.com/qaz812345/TrackNetV3 as a sibling
folder to this `phase1` directory and downloaded the pretrained weights.

Expected layout:

    Line Calls/
    ├── phase1/                      <-- you are here
    │   ├── tracknet_detector.py
    │   ├── process.py
    │   └── ...
    └── TrackNetV3/                  <-- cloned repo
        ├── model.py                 (or similar — see TRACKNET_MODEL_IMPORT below)
        └── ckpts/
            └── TrackNetV3_best.pt   (or whatever weights file you downloaded)

The four configuration constants below are the ONLY things you should need to
adjust to match the real TrackNetV3 repo on disk. If anything else breaks,
read the error message — the wrapper is defensive about reporting what it
expected vs. what it found.

------------------------------------------------------------
TRACKNET INTEGRATION POINTS (likely to need a small tweak)
------------------------------------------------------------

1) TRACKNET_REPO_PATH    - where you cloned the repo
2) TRACKNET_WEIGHTS      - path to the .pt / .pth file
3) TRACKNET_MODEL_IMPORT - Python import for the model class
4) MODEL_KWARGS          - kwargs the model class wants (in_dim, out_dim)
   INPUT_HEIGHT / INPUT_WIDTH - input resolution the model expects

If your TrackNetV3 fork exposes a `predict_one(...)` or `infer(...)` function,
you can short-circuit and call it directly from `detect_video` instead of
running our own forward pass. The point of this wrapper is to give you one
clean interface (`detect_video(path) -> dict[int, dict]`) to plug into
process.py. Everything downstream uses the same shape.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as e:
    raise ImportError(
        "PyTorch is required for the TrackNet detector. "
        "Install it with `pip install -r requirements.txt`, or follow "
        "https://pytorch.org/get-started/locally/ for a CUDA build."
    ) from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731


# ===========================================================================
# CONFIGURATION — confirmed against the official TrackNetV3 checkpoint
# (qaz812345/TrackNetV3, checkpoint TrackNet_best.pt, Aug 2023)
# ===========================================================================

# 1) Where you cloned qaz812345/TrackNetV3 (sibling to phase1 by default)
TRACKNET_REPO_PATH = Path(
    os.environ.get("TRACKNET_REPO_PATH",
                   str(Path(__file__).resolve().parent.parent / "TrackNetV3"))
)

# 2) Path to the pretrained weights you downloaded.
#    The official zip extracts as TrackNet_best.pt (not TrackNetV3_best.pt).
TRACKNET_WEIGHTS = Path(
    os.environ.get("TRACKNET_WEIGHTS",
                   str(TRACKNET_REPO_PATH / "ckpts" / "TrackNet_best.pt"))
)

# 3) Import path for the model class.
#    The repo defines class TrackNet in model.py (not TrackNetV3).
TRACKNET_MODEL_IMPORT = os.environ.get(
    "TRACKNET_MODEL_IMPORT", "model:TrackNet"
)

# 4) Model constructor kwargs.
#    Checkpoint param_dict: seq_len=8, bg_mode='concat'
#    → in_dim = (8+1)*3 = 27  (8 RGB frames + 1 background RGB frame)
#    → out_dim = 8             (one heatmap per input frame)
MODEL_KWARGS = {"in_dim": 27, "out_dim": 8}

# Number of consecutive frames the model sees per forward pass.
SEQ_LEN = 8  # must match MODEL_KWARGS above

# Input resolution the model was trained on.
INPUT_HEIGHT = 288
INPUT_WIDTH = 512

# How many frames to sample when estimating the background median.
# Higher = more accurate background, slower startup.
BG_SAMPLE_FRAMES = 200

# Heatmap peak threshold for a detection to count as valid.
# Lower = more detections (and more false positives).
CONFIDENCE_THRESHOLD = 0.3

# When loading the checkpoint, common wrapping keys to unwrap.
STATE_DICT_KEYS_TO_TRY = ["model", "state_dict", "model_state_dict", "net"]


# ===========================================================================
# Detection record
# ===========================================================================

@dataclass
class FrameDetection:
    frame: int
    t: float
    x: float
    y: float
    confidence: float


# ===========================================================================
# Model loading
# ===========================================================================

def _load_model(device: str):
    """Import the TrackNetV3 model class from the cloned repo and load weights."""
    if not TRACKNET_REPO_PATH.exists():
        raise FileNotFoundError(
            f"TrackNetV3 repo not found at {TRACKNET_REPO_PATH}. "
            f"Clone it: git clone https://github.com/qaz812345/TrackNetV3.git "
            f"into the parent of phase1/, or set TRACKNET_REPO_PATH env var."
        )
    if not TRACKNET_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found at {TRACKNET_WEIGHTS}. "
            f"Download the pretrained .pt file per the TrackNetV3 README, "
            f"or set TRACKNET_WEIGHTS env var."
        )

    sys.path.insert(0, str(TRACKNET_REPO_PATH))

    if ":" not in TRACKNET_MODEL_IMPORT:
        raise ValueError(
            f"TRACKNET_MODEL_IMPORT must be in 'module:ClassName' form, "
            f"got {TRACKNET_MODEL_IMPORT!r}"
        )
    module_name, class_name = TRACKNET_MODEL_IMPORT.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import '{module_name}' from {TRACKNET_REPO_PATH}. "
            f"Adjust TRACKNET_MODEL_IMPORT or check the repo layout. "
            f"Original error: {e}"
        ) from e
    if not hasattr(module, class_name):
        attrs = [a for a in dir(module) if not a.startswith("_")]
        raise AttributeError(
            f"Module '{module_name}' has no class '{class_name}'. "
            f"Found these top-level names: {attrs}. "
            f"Adjust TRACKNET_MODEL_IMPORT."
        )
    ModelClass = getattr(module, class_name)

    print(f"Loading {class_name} from {module_name} on {device}...")
    try:
        model = ModelClass(**MODEL_KWARGS)
    except TypeError as e:
        raise TypeError(
            f"{class_name}{tuple(MODEL_KWARGS.items())} failed: {e}. "
            f"Adjust MODEL_KWARGS for your TrackNetV3 fork."
        ) from e

    raw = torch.load(str(TRACKNET_WEIGHTS), map_location=device)
    state_dict = raw
    if isinstance(raw, dict):
        for k in STATE_DICT_KEYS_TO_TRY:
            if k in raw and isinstance(raw[k], dict):
                state_dict = raw[k]
                break

    # Strip "module." prefix if checkpoint was saved from DataParallel
    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  warning: missing keys when loading weights: {len(missing)}")
    if unexpected:
        print(f"  warning: unexpected keys when loading weights: {len(unexpected)}")

    model.to(device).eval()
    return model


# ===========================================================================
# Pre/postprocessing
# ===========================================================================

def _compute_background(video_path: str) -> np.ndarray:
    """
    Estimate a background frame by taking the per-pixel median of a sample of
    frames from the video.  Returns a float32 RGB array of shape
    (INPUT_HEIGHT, INPUT_WIDTH, 3) in [0, 1].
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // BG_SAMPLE_FRAMES) if total > 0 else 1

    frames_rgb = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rs = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT),
                            interpolation=cv2.INTER_LINEAR)
            frames_rgb.append(rs.astype(np.float32) / 255.0)
        idx += 1
    cap.release()

    if not frames_rgb:
        # Fallback: grey frame
        return np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 0.5, dtype=np.float32)

    print(f"  Background: median of {len(frames_rgb)} sampled frames.")
    return np.median(np.stack(frames_rgb, axis=0), axis=0).astype(np.float32)


def _resize_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Resize a BGR frame to (INPUT_HEIGHT, INPUT_WIDTH) and return float32 RGB in [0,1]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rs = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return rs.astype(np.float32) / 255.0


def _preprocess_sequence(bg: np.ndarray,
                         seq: list[np.ndarray]) -> torch.Tensor:
    """
    Build the model input tensor for one window of SEQ_LEN frames.

    bg:  float32 RGB (INPUT_HEIGHT, INPUT_WIDTH, 3) in [0, 1]  — background median
    seq: list of SEQ_LEN float32 RGB frames in [0, 1]

    Returns: tensor of shape (1, (SEQ_LEN+1)*3, INPUT_HEIGHT, INPUT_WIDTH)
    Channel layout (matching bg_mode='concat' in the official repo):
        channels  0– 2 : background frame
        channels  3– 5 : seq[0]  (oldest)
        channels  6– 8 : seq[1]
        ...
        channels 24–26 : seq[7]  (newest / "current")
    """
    assert len(seq) == SEQ_LEN, f"Expected {SEQ_LEN} frames, got {len(seq)}"
    parts = [bg] + list(seq)                        # 9 × (H, W, 3)
    stacked = np.concatenate(parts, axis=2)         # (H, W, 27)
    tensor = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor


def _peak_from_heatmap(heatmap: np.ndarray) -> tuple[float, float, float] | None:
    """
    heatmap: (H, W) numpy array of confidences in [0, 1]-ish (may be logits
             before sigmoid; we normalize to be safe).
    Returns (x, y, conf) in heatmap coordinates, or None if below threshold.
    """
    # Normalize: if values look like logits, sigmoid them
    hmax = float(heatmap.max())
    if hmax > 1.5:  # heuristic: logits, not probabilities
        heatmap = 1.0 / (1.0 + np.exp(-heatmap))
        hmax = float(heatmap.max())
    if hmax < CONFIDENCE_THRESHOLD:
        return None
    # Subpixel peak via parabolic fit on a 3x3 window
    iy, ix = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    h, w = heatmap.shape
    sub_y = float(iy)
    sub_x = float(ix)
    if 1 <= iy < h - 1 and 1 <= ix < w - 1:
        # Refine y
        y0, y1, y2 = heatmap[iy-1, ix], heatmap[iy, ix], heatmap[iy+1, ix]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) > 1e-6:
            sub_y = iy + 0.5 * (y0 - y2) / denom
        # Refine x
        x0, x1, x2 = heatmap[iy, ix-1], heatmap[iy, ix], heatmap[iy, ix+1]
        denom = (x0 - 2*x1 + x2)
        if abs(denom) > 1e-6:
            sub_x = ix + 0.5 * (x0 - x2) / denom
    return sub_x, sub_y, hmax


def _rescale_to_original(x_in: float, y_in: float,
                         orig_w: int, orig_h: int) -> tuple[float, float]:
    return (x_in * orig_w / INPUT_WIDTH,
            y_in * orig_h / INPUT_HEIGHT)


# ===========================================================================
# Public API
# ===========================================================================

class TrackNetBallDetector:
    """
    Tennis ball detector backed by TrackNetV3.

    Usage:
        det = TrackNetBallDetector()                # auto-picks device
        det.detect_video("test_rally.mp4")          # -> dict[frame_idx -> {...}]
    """

    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = _load_model(device)
        print(f"TrackNetV3 ready on {device} "
              f"(seq_len={SEQ_LEN}, bg_mode=concat, "
              f"input {INPUT_WIDTH}x{INPUT_HEIGHT}, "
              f"threshold {CONFIDENCE_THRESHOLD}).")

    @torch.no_grad()
    def detect_video(self, video_path: str) -> dict[int, dict]:
        """
        Run TrackNetV3 inference over an entire video.

        The official model uses seq_len=8 + bg_mode='concat':
          - First pass: compute a background median frame from a frame sample.
          - Second pass: slide a window of SEQ_LEN frames over the video
            (non-overlapping stride = SEQ_LEN).  Each forward pass yields
            SEQ_LEN heatmaps; we extract a detection from each.

        Returns a dict with keys:
            fps, width, height, total_frames,
            detections: {frame_idx -> {frame, t, x, y, confidence}}
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        # ---- Pass 1: background estimation ----
        print("Computing background median (pass 1)...")
        bg = _compute_background(video_path)           # (H, W, 3) float32

        # ---- Pass 2: read all frames into a list ----
        # For Phase 1 proof-of-concept, videos are short (30 s × 240 fps ≈ 7200
        # frames).  Loading everything avoids reopening the file repeatedly.
        # If memory is tight, drop this and re-read from cap in the loop below.
        print("Reading frames (pass 2)...")
        cap = cv2.VideoCapture(video_path)
        all_frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            all_frames.append(_resize_frame(frame))
        cap.release()

        n_frames = len(all_frames)
        print(f"  {n_frames} frames at {fps:.1f} fps "
              f"({orig_w}×{orig_h} → {INPUT_WIDTH}×{INPUT_HEIGHT})")

        # ---- Pass 3: sliding-window inference ----
        results: dict[int, dict] = {}
        first_output_logged = False

        # Pad the frame list so the last window is always full
        pad_len = (SEQ_LEN - n_frames % SEQ_LEN) % SEQ_LEN
        padded = all_frames + [all_frames[-1]] * pad_len if all_frames else []

        n_windows = len(padded) // SEQ_LEN
        progress = tqdm(total=n_windows, desc="TrackNet windows", unit="win")

        for w in range(n_windows):
            start = w * SEQ_LEN
            seq = padded[start: start + SEQ_LEN]

            tensor = _preprocess_sequence(bg, seq).to(self.device)
            out = self.model(tensor)

            if isinstance(out, (tuple, list)):
                out = out[0]
            if not first_output_logged:
                print(f"  TrackNet output shape: {tuple(out.shape)}")
                first_output_logged = True

            # out shape: (1, SEQ_LEN, H, W)
            if out.dim() != 4:
                raise RuntimeError(
                    f"Unexpected TrackNet output shape {tuple(out.shape)}. "
                    f"Expected (1, {SEQ_LEN}, H, W). "
                    f"Edit tracknet_detector.py to handle this case."
                )

            for i in range(SEQ_LEN):
                frame_idx = start + i
                if frame_idx >= n_frames:
                    break   # skip padding frames

                heatmap = out[0, i].cpu().numpy()
                peak = _peak_from_heatmap(heatmap)
                if peak is not None:
                    px_in, py_in, conf = peak
                    x_orig, y_orig = _rescale_to_original(px_in, py_in,
                                                          orig_w, orig_h)
                    results[frame_idx] = {
                        "frame": frame_idx,
                        "t": frame_idx / fps,
                        "x": x_orig,
                        "y": y_orig,
                        "confidence": conf,
                    }

            progress.update(1)

        progress.close()
        print(f"TrackNet detected ball in {len(results)} / {n_frames} frames "
              f"({100.0 * len(results) / max(1, n_frames):.1f}%).")
        return {
            "fps": fps,
            "width": orig_w,
            "height": orig_h,
            "total_frames": n_frames,
            "detections": results,
        }
