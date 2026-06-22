"""
Shared court geometry and homography helpers.

Coordinate convention:
- Court coordinates are in METERS.
- Origin (0, 0) is at the NEAR-LEFT singles court corner (closest to the camera, on the left).
- X axis: 0 -> 8.23 m, running along the near baseline toward the right.
- Y axis: 0 -> 23.77 m, running down the court toward the far baseline.

Singles court dimensions (ITF):
- Length:  23.77 m (baseline to baseline)
- Width:    8.23 m (singles sideline to singles sideline)
- Service line: 6.40 m from net on each side -> y = 5.485 and y = 18.285
- Net at:   y = 11.885

Tennis ball radius: 0.0335 m (officially 0.0327-0.0343 m, average ~33.5 mm).
"""

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Court dimensions (singles, ITF spec)
# ---------------------------------------------------------------------------
COURT_LENGTH = 23.77     # meters, baseline to baseline
COURT_WIDTH  = 8.23      # meters, singles sideline to singles sideline
SERVICE_LINE_FROM_NET = 6.40
NET_Y = COURT_LENGTH / 2  # 11.885 m
NEAR_SERVICE_Y = NET_Y - SERVICE_LINE_FROM_NET   # 5.485 m
FAR_SERVICE_Y  = NET_Y + SERVICE_LINE_FROM_NET   # 18.285 m
CENTER_X = COURT_WIDTH / 2  # 4.115 m

BALL_RADIUS = 0.0335  # meters

# Real-world singles corner positions in the same order the calibration UI asks for them:
# 1. near-left, 2. near-right, 3. far-right, 4. far-left
CORNER_POSITIONS_M = np.array([
    [0.0,          0.0],          # near-left
    [COURT_WIDTH,  0.0],          # near-right
    [COURT_WIDTH,  COURT_LENGTH], # far-right
    [0.0,          COURT_LENGTH], # far-left
], dtype=np.float64)


def compute_homography(image_corners_px):
    """
    image_corners_px: 4x2 array of pixel coordinates in the order
        [near-left, near-right, far-right, far-left].
    Returns: 3x3 homography matrix mapping pixel coords -> court meters.
    """
    src = np.asarray(image_corners_px, dtype=np.float64)
    dst = CORNER_POSITIONS_M
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RuntimeError("Homography computation failed. Did you click 4 distinct corners?")
    return H


def pixel_to_court(H, px, py):
    """Apply homography to a single (px, py) pixel point. Returns (x_m, y_m)."""
    v = np.array([px, py, 1.0], dtype=np.float64)
    w = H @ v
    if abs(w[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(w[0] / w[2]), float(w[1] / w[2])


def court_to_pixel(H, x_m, y_m):
    """Inverse: project a court (x, y) meter point back to image pixels."""
    H_inv = np.linalg.inv(H)
    v = np.array([x_m, y_m, 1.0], dtype=np.float64)
    w = H_inv @ v
    if abs(w[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(w[0] / w[2]), float(w[1] / w[2])


# ---------------------------------------------------------------------------
# In/out decision
# ---------------------------------------------------------------------------
def classify_bounce(x_m, y_m, ball_radius=BALL_RADIUS):
    """
    Decide whether a bounce at court coords (x_m, y_m) is IN, OUT, or CLOSE
    for the singles court.

    A ball is considered IN if any part of it touches the line. The bounce
    center must therefore be within `ball_radius` outside the line to still
    be IN.

    Returns: dict with keys 'call', 'reason', 'margin_m', 'confidence_band'.
    """
    # Margins (positive = inside, negative = outside)
    margin_left   = x_m
    margin_right  = COURT_WIDTH - x_m
    margin_near   = y_m
    margin_far    = COURT_LENGTH - y_m

    margins = {
        "left sideline":  margin_left,
        "right sideline": margin_right,
        "near baseline":  margin_near,
        "far baseline":   margin_far,
    }

    # Worst (most negative or smallest positive) margin
    worst_line, worst_margin = min(margins.items(), key=lambda kv: kv[1])

    # Decision
    if worst_margin >= ball_radius:
        call = "IN"
        reason = f"clear by {worst_margin*100:.1f}cm at {worst_line}"
    elif worst_margin >= -ball_radius:
        # Within a ball-radius of the line -> ball touches the line -> IN, but mark it close
        call = "IN"
        reason = f"close (touching {worst_line}, {worst_margin*100:+.1f}cm)"
    else:
        call = "OUT"
        outside_by = -worst_margin - ball_radius
        reason = f"{outside_by*100:.1f}cm past {worst_line}"

    return {
        "call": call,
        "reason": reason,
        "margin_m": worst_margin,
        "worst_line": worst_line,
    }
