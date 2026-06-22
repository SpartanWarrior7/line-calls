"""
Process a tennis video to detect the ball (TrackNetV3), find bounces,
and classify in/out using the homography from calibrate.py.

Usage:
    python process.py --video test_rally.mp4 --calibration court_calibration.json

Output:
    results.json -- per-frame detections, trajectories, bounces, calls.
                    Same schema as before, so visualize.py keeps working.

Pipeline:
    1. Run TrackNetV3 across the video      -> per-frame ball (x, y, conf)
    2. Group consecutive detections          -> trajectories
       (split when the gap exceeds MAX_TRAJECTORY_GAP_FRAMES)
    3. Find bounces per trajectory           -> local maxima of image-y
                                                with positive->negative
                                                vy sign change
    4. Project bounces to court meters       -> via homography
    5. Classify in/out                       -> courtmath.classify_bounce
"""
import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from courtmath import pixel_to_court, classify_bounce
from tracknet_detector import TrackNetBallDetector


# ===========================================================================
# Tunables
# ===========================================================================

# Largest frame gap (in frames) between two consecutive detections that we
# still glue into a single trajectory. Anything bigger starts a new one.
MAX_TRAJECTORY_GAP_FRAMES = 8

# Minimum number of detections required for a trajectory to be considered
# real (filters out one-off false positives from TrackNet).
MIN_TRAJECTORY_LENGTH = 6

# Bounce detection: smoothing window (in frames) over the image-y values.
# At 240 fps a window of 5 is only ~20 ms — not enough to suppress jitter.
# Raise to 9 at 240 fps; keep at 5 for 60 fps. Override with --fps-hint if
# your source is unusual.
BOUNCE_SMOOTH_WINDOW = 3

# Minimum separation (in frames) between two consecutive bounce detections
# in the same trajectory. Prevents double-counting a single bounce event.
MIN_BOUNCE_SEPARATION_FRAMES = 12

# Minimum drop in vertical velocity across a candidate bounce frame
# (pixels-per-frame). Rejects plateaus that aren't real bounces.
MIN_BOUNCE_VELOCITY_CHANGE = 3.0

# How close to the net (in court meters) a projected bounce must be before
# we reject it as a net-crossing artifact.  The net is at y = 11.885 m;
# no real bounce lands within this band of it.
NET_ZONE_BUFFER_M = 1.5   # rejects bounces between 10.4 m and 13.4 m

# Bounces that project outside the court by more than this tolerance are
# discarded as calibration / detection errors (not real bounces).
COURT_MARGIN_TOLERANCE_M = 0.5

# Camera side: "near" means the camera is behind the NEAR baseline (y ≈ 0).
# When True, near-side bounces are local MAXIMA of image-y and far-side
# bounces are local MINIMA.  Flip to False if the camera is at the far end.
CAMERA_AT_NEAR_BASELINE = True


# ===========================================================================
# Data types
# ===========================================================================

@dataclass
class Detection:
    frame: int
    t: float
    x: float
    y: float
    confidence: float


@dataclass
class Trajectory:
    detections: list[Detection] = field(default_factory=list)


@dataclass
class Bounce:
    trajectory_idx: int
    frame: int
    t: float
    x_px: float
    y_px: float
    x_court_m: float
    y_court_m: float
    call: str
    reason: str
    margin_m: float


# ===========================================================================
# Trajectory assembly
# ===========================================================================

def group_into_trajectories(detections_by_frame: dict[int, dict],
                            fps: float) -> list[Trajectory]:
    """
    Walk through the per-frame detections in frame order and split into
    trajectories whenever the gap between consecutive detections exceeds
    MAX_TRAJECTORY_GAP_FRAMES.
    """
    if not detections_by_frame:
        return []

    frames_sorted = sorted(detections_by_frame.keys())
    trajectories: list[Trajectory] = []
    current = Trajectory()
    prev_frame = None

    for f in frames_sorted:
        d = detections_by_frame[f]
        if prev_frame is not None and (f - prev_frame) > MAX_TRAJECTORY_GAP_FRAMES:
            if len(current.detections) >= MIN_TRAJECTORY_LENGTH:
                trajectories.append(current)
            current = Trajectory()
        current.detections.append(Detection(
            frame=f,
            t=f / fps,
            x=float(d["x"]),
            y=float(d["y"]),
            confidence=float(d.get("confidence", 1.0)),
        ))
        prev_frame = f

    if len(current.detections) >= MIN_TRAJECTORY_LENGTH:
        trajectories.append(current)
    return trajectories


# ===========================================================================
# Bounce detection
# ===========================================================================

def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.astype(float).copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def find_bounces(traj: Trajectory) -> list[int]:
    """
    Return indices (into traj.detections) where a bounce occurred.

    Detects BOTH near-side and far-side bounces:

    - Near-side bounce (camera behind near baseline):
        local MAXIMUM of image-y  (vy: + → −, ball was moving DOWN then UP)
    - Far-side bounce:
        local MINIMUM of image-y  (vy: − → +, ball was moving UP then DOWN)

    After finding raw candidates we deduplicate to prevent double-counting
    a single bounce event (MIN_BOUNCE_SEPARATION_FRAMES).
    """
    if len(traj.detections) < 5:
        return []

    ys = np.array([d.y for d in traj.detections], dtype=float)
    ys_smooth = _smooth(ys, BOUNCE_SMOOTH_WINDOW)
    vy = np.diff(ys_smooth)

    candidates = []
    for i in range(1, len(vy)):
        dv = abs(vy[i - 1] - vy[i])
        if dv < MIN_BOUNCE_VELOCITY_CHANGE:
            continue

        if CAMERA_AT_NEAR_BASELINE:
            # Near-side: local max of y (vy: positive → negative)
            if vy[i - 1] > 0 and vy[i] < 0:
                candidates.append(i)
            # Far-side: local min of y (vy: negative → positive)
            elif vy[i - 1] < 0 and vy[i] > 0:
                candidates.append(i)
        else:
            # Camera at far baseline — flip the logic
            if vy[i - 1] < 0 and vy[i] > 0:
                candidates.append(i)
            elif vy[i - 1] > 0 and vy[i] < 0:
                candidates.append(i)

    # Deduplicate: if two candidates are within MIN_BOUNCE_SEPARATION_FRAMES
    # of each other, keep only the one with the larger velocity change.
    if not candidates:
        return []

    frame_indices = [traj.detections[i].frame for i in candidates]
    dv_scores = [abs(vy[i - 1] - vy[i]) for i in candidates]

    kept = []
    used = [False] * len(candidates)
    for a in range(len(candidates)):
        if used[a]:
            continue
        group = [a]
        for b in range(a + 1, len(candidates)):
            if not used[b] and abs(frame_indices[b] - frame_indices[a]) <= MIN_BOUNCE_SEPARATION_FRAMES:
                group.append(b)
                used[b] = True
        best = max(group, key=lambda k: dv_scores[k])
        kept.append(candidates[best])

    return sorted(kept)


# ===========================================================================
# Main pipeline
# ===========================================================================

def process(video_path: str, calibration_path: str, output_path: str,
            device: str | None = None):
    calib = json.loads(Path(calibration_path).read_text())
    H = np.array(calib["homography_pixel_to_court_m"], dtype=np.float64)

    # ---- Step 1: TrackNet inference over the whole video ----
    detector = TrackNetBallDetector(device=device)
    det_payload = detector.detect_video(video_path)
    fps    = det_payload["fps"]
    width  = det_payload["width"]
    height = det_payload["height"]
    total_frames = det_payload["total_frames"]
    detections_by_frame = det_payload["detections"]

    # ---- Step 2: group into trajectories ----
    trajectories = group_into_trajectories(detections_by_frame, fps)
    print(f"Assembled {len(trajectories)} trajectories "
          f"(min length {MIN_TRAJECTORY_LENGTH}).")

    from courtmath import NET_Y, COURT_LENGTH, COURT_WIDTH

    # ---- Steps 3 & 4: find bounces and project to court ----
    all_bounces: list[Bounce] = []
    rejected_count = 0
    for traj_idx, traj in enumerate(trajectories):
        bounce_indices = find_bounces(traj)
        for bi in bounce_indices:
            d = traj.detections[bi]
            x_m, y_m = pixel_to_court(H, d.x, d.y)

            # ---- Plausibility filter 1: reject net-zone artifacts ----
            # The net is at y = NET_Y m.  Nothing real bounces within
            # NET_ZONE_BUFFER_M of it.
            if abs(y_m - NET_Y) < NET_ZONE_BUFFER_M:
                rejected_count += 1
                print(f"  [skip] net-zone artifact at t={d.t:.2f}s "
                      f"court=({x_m:.2f}, {y_m:.2f})")
                continue

            # ---- Plausibility filter 2: reject wildly out-of-court points ----
            # If the projected position is more than COURT_MARGIN_TOLERANCE_M
            # outside any edge, it's a detection error, not a real bounce.
            tol = COURT_MARGIN_TOLERANCE_M
            if (x_m < -tol or x_m > COURT_WIDTH + tol or
                    y_m < -tol or y_m > COURT_LENGTH + tol):
                rejected_count += 1
                print(f"  [skip] out-of-bounds projection at t={d.t:.2f}s "
                      f"court=({x_m:.2f}, {y_m:.2f})")
                continue

            result = classify_bounce(x_m, y_m)
            all_bounces.append(Bounce(
                trajectory_idx=traj_idx,
                frame=d.frame,
                t=d.t,
                x_px=d.x,
                y_px=d.y,
                x_court_m=x_m,
                y_court_m=y_m,
                call=result["call"],
                reason=result["reason"],
                margin_m=result["margin_m"],
            ))

    if rejected_count:
        print(f"  ({rejected_count} candidate bounce(s) rejected by plausibility filters)")

    # ---- Print summary ----
    print(f"\nDetected {len(all_bounces)} bounces:")
    for i, b in enumerate(all_bounces, 1):
        print(f"  #{i:2d}  t={b.t:6.2f}s  court=({b.x_court_m:6.2f}m, {b.y_court_m:6.2f}m)  "
              f"-> {b.call:3s}  ({b.reason})")

    # ---- Save results.json (schema unchanged from prior version) ----
    payload = {
        "video": str(Path(video_path).resolve()),
        "calibration": str(Path(calibration_path).resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "detector": "TrackNetV3",
        "trajectories": [
            {"detections": [asdict(d) for d in t.detections]}
            for t in trajectories
        ],
        "bounces": [asdict(b) for b in all_bounces],
        "parameters": {
            "MAX_TRAJECTORY_GAP_FRAMES": MAX_TRAJECTORY_GAP_FRAMES,
            "MIN_TRAJECTORY_LENGTH": MIN_TRAJECTORY_LENGTH,
            "BOUNCE_SMOOTH_WINDOW": BOUNCE_SMOOTH_WINDOW,
            "MIN_BOUNCE_SEPARATION_FRAMES": MIN_BOUNCE_SEPARATION_FRAMES,
            "MIN_BOUNCE_VELOCITY_CHANGE": MIN_BOUNCE_VELOCITY_CHANGE,
            "NET_ZONE_BUFFER_M": NET_ZONE_BUFFER_M,
            "COURT_MARGIN_TOLERANCE_M": COURT_MARGIN_TOLERANCE_M,
            "CAMERA_AT_NEAR_BASELINE": CAMERA_AT_NEAR_BASELINE,
        },
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--calibration", default="court_calibration.json")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu | mps. Auto-detected if omitted.")
    args = parser.parse_args()
    process(args.video, args.calibration, args.output, args.device)


if __name__ == "__main__":
    main()
