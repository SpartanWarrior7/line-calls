"""
Render visualizations from process.py's results.

Produces:
    overlay.mp4   - original video with ball trail, bounce markers,
                    court lines drawn from homography, and in/out call text
    topdown.png   - a top-down singles-court diagram with all bounce
                    points plotted

Usage:
    python visualize.py --video test_rally.mp4 \
                        --results results.json \
                        --calibration court_calibration.json
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from courtmath import (
    court_to_pixel,
    COURT_LENGTH,
    COURT_WIDTH,
    NEAR_SERVICE_Y,
    FAR_SERVICE_Y,
    CENTER_X,
)


# How many recent detections to show in the ball trail
TRAIL_LENGTH = 18

# How many frames to highlight a bounce after it occurs
BOUNCE_FLASH_FRAMES = 40


def draw_court(img, H, color=(0, 255, 0)):
    def p(x, y):
        px, py = court_to_pixel(H, x, y)
        return int(px), int(py)
    # Singles boundary
    cv2.line(img, p(0, 0),               p(COURT_WIDTH, 0),               color, 2)
    cv2.line(img, p(COURT_WIDTH, 0),     p(COURT_WIDTH, COURT_LENGTH),    color, 2)
    cv2.line(img, p(COURT_WIDTH, COURT_LENGTH), p(0, COURT_LENGTH),       color, 2)
    cv2.line(img, p(0, COURT_LENGTH),    p(0, 0),                         color, 2)
    # Service lines
    cv2.line(img, p(0, NEAR_SERVICE_Y),  p(COURT_WIDTH, NEAR_SERVICE_Y),  color, 1)
    cv2.line(img, p(0, FAR_SERVICE_Y),   p(COURT_WIDTH, FAR_SERVICE_Y),   color, 1)
    cv2.line(img, p(CENTER_X, NEAR_SERVICE_Y), p(CENTER_X, FAR_SERVICE_Y), color, 1)
    # Net
    cv2.line(img, p(0, COURT_LENGTH/2),  p(COURT_WIDTH, COURT_LENGTH/2),  (0, 200, 200), 1)


def color_for_call(call: str) -> tuple[int, int, int]:
    if call == "IN":
        return (0, 200, 0)
    if call == "OUT":
        return (0, 0, 220)
    return (0, 200, 220)


def render_overlay(video_path: str, results_path: str, calibration_path: str,
                   output_path: str):
    results = json.loads(Path(results_path).read_text())
    calib   = json.loads(Path(calibration_path).read_text())
    H = np.array(calib["homography_pixel_to_court_m"], dtype=np.float64)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = results.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = results.get("width")  or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = results.get("height") or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {output_path}")

    # Build a per-frame index of detections (across all trajectories)
    # and a list of bounces.
    detections_by_frame: dict[int, list[tuple[float, float]]] = {}
    for traj in results["trajectories"]:
        for d in traj["detections"]:
            detections_by_frame.setdefault(d["frame"], []).append((d["x"], d["y"]))

    bounces = results["bounces"]
    bounce_frames = {b["frame"]: b for b in bounces}

    # Pre-build trails: for each frame, the list of recent ball positions
    # within TRAIL_LENGTH frames (from same trajectory).
    trail_lookup: dict[int, list[tuple[float, float]]] = {}
    for traj in results["trajectories"]:
        dets = traj["detections"]
        for i, d in enumerate(dets):
            start = max(0, i - TRAIL_LENGTH)
            trail_lookup[d["frame"]] = [(dd["x"], dd["y"]) for dd in dets[start:i+1]]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_court(frame, H, color=(0, 255, 0))

        # Trail
        trail = trail_lookup.get(frame_idx)
        if trail:
            for i in range(1, len(trail)):
                x1, y1 = trail[i-1]
                x2, y2 = trail[i]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 255, 255), 2)
            x, y = trail[-1]
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), 2)

        # Recent bounces: flash a colored ring on the bounce point for a bit
        for b in bounces:
            if 0 <= frame_idx - b["frame"] <= BOUNCE_FLASH_FRAMES:
                color = color_for_call(b["call"])
                cv2.circle(frame, (int(b["x_px"]), int(b["y_px"])), 14, color, 3)
                label = f"{b['call']}  ({b['reason']})"
                cv2.putText(frame, label,
                            (int(b["x_px"]) + 18, int(b["y_px"]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # HUD: frame count, timestamp
        cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {frame_idx}  t={frame_idx/fps:.2f}s  "
                           f"bounces={len(bounces)}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved overlay video to {output_path}")


def render_topdown(results_path: str, output_path: str,
                   px_per_m: int = 60):
    results = json.loads(Path(results_path).read_text())
    bounces = results["bounces"]

    # Image dimensions: court + margin
    margin_m = 1.5
    img_w = int((COURT_WIDTH  + 2 * margin_m) * px_per_m)
    img_h = int((COURT_LENGTH + 2 * margin_m) * px_per_m)
    img = np.full((img_h, img_w, 3), 245, dtype=np.uint8)  # light gray bg

    def to_px(x_m, y_m):
        return (int((x_m + margin_m) * px_per_m),
                int((y_m + margin_m) * px_per_m))

    # Court fill
    cv2.rectangle(img,
                  to_px(0, 0), to_px(COURT_WIDTH, COURT_LENGTH),
                  (180, 130, 90), -1)

    # Court lines (white)
    line_color = (255, 255, 255)
    cv2.rectangle(img, to_px(0, 0), to_px(COURT_WIDTH, COURT_LENGTH),
                  line_color, 2)
    cv2.line(img, to_px(0, NEAR_SERVICE_Y), to_px(COURT_WIDTH, NEAR_SERVICE_Y),
             line_color, 2)
    cv2.line(img, to_px(0, FAR_SERVICE_Y),  to_px(COURT_WIDTH, FAR_SERVICE_Y),
             line_color, 2)
    cv2.line(img, to_px(CENTER_X, NEAR_SERVICE_Y),
             to_px(CENTER_X, FAR_SERVICE_Y), line_color, 2)
    # Net
    cv2.line(img, to_px(0, COURT_LENGTH/2), to_px(COURT_WIDTH, COURT_LENGTH/2),
             (40, 40, 40), 2)

    # Compass / scale
    cv2.putText(img, "near baseline (camera side)",
                to_px(0.1, -0.3), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "far baseline",
                to_px(0.1, COURT_LENGTH + 0.9), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Bounce points
    for i, b in enumerate(bounces, 1):
        color = color_for_call(b["call"])
        # Convert call-color from BGR to BGR (already BGR — no-op, just clarifying)
        center = to_px(b["x_court_m"], b["y_court_m"])
        cv2.circle(img, center, 8, color, -1)
        cv2.circle(img, center, 8, (0, 0, 0), 1)
        cv2.putText(img, str(i), (center[0] + 10, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Legend
    legend_y = 30
    cv2.putText(img, "IN",  (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "OUT", (60, legend_y), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 220), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Saved top-down diagram to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--calibration", default="court_calibration.json")
    parser.add_argument("--overlay-out", default="overlay.mp4")
    parser.add_argument("--topdown-out", default="topdown.png")
    parser.add_argument("--skip-overlay", action="store_true",
                        help="Skip the overlay video (faster if you only want the top-down).")
    args = parser.parse_args()

    if not args.skip_overlay:
        render_overlay(args.video, args.results, args.calibration, args.overlay_out)
    render_topdown(args.results, args.topdown_out)


if __name__ == "__main__":
    main()
