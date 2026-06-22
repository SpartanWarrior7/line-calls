"""
Interactive calibration tool.

Opens the first frame of your video. Click the four singles court corners in
this order:

    1. NEAR-LEFT   (closest to camera, left side)
    2. NEAR-RIGHT
    3. FAR-RIGHT
    4. FAR-LEFT

Keys:
    r       - reset clicks
    Enter   - accept (only enabled once 4 points are placed)
    q / Esc - quit without saving

Output: court_calibration.json (or --output PATH)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from courtmath import compute_homography, court_to_pixel, COURT_LENGTH, COURT_WIDTH


CORNER_LABELS = ["1: NEAR-LEFT", "2: NEAR-RIGHT", "3: FAR-RIGHT", "4: FAR-LEFT"]


def grab_frame(video_path: str, frame_idx: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def draw_overlay(frame, points):
    out = frame.copy()
    h, w = out.shape[:2]

    # Instructions banner
    next_label = CORNER_LABELS[len(points)] if len(points) < 4 else "Press Enter to save"
    banner = f"Click corner {next_label}   |   r=reset  Enter=save  Esc=quit  ({len(points)}/4)"
    cv2.rectangle(out, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Drawn points
    for i, (x, y) in enumerate(points):
        color = (0, 255, 255)
        cv2.circle(out, (int(x), int(y)), 8, color, 2)
        cv2.circle(out, (int(x), int(y)), 2, color, -1)
        cv2.putText(out, CORNER_LABELS[i], (int(x) + 12, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # If we have all four, draw the implied court so you can sanity-check
    if len(points) == 4:
        try:
            H = compute_homography(np.array(points, dtype=np.float64))
            draw_court_overlay(out, H)
        except Exception as e:
            cv2.putText(out, f"Bad calibration: {e}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    return out


def draw_court_overlay(img, H, color=(0, 255, 0)):
    """Draw the singles court lines back onto the image, using the homography."""
    def p(x, y):
        px, py = court_to_pixel(H, x, y)
        return int(px), int(py)

    # Baselines
    cv2.line(img, p(0, 0),               p(COURT_WIDTH, 0),               color, 2)
    cv2.line(img, p(0, COURT_LENGTH),    p(COURT_WIDTH, COURT_LENGTH),    color, 2)
    # Sidelines
    cv2.line(img, p(0, 0),               p(0, COURT_LENGTH),              color, 2)
    cv2.line(img, p(COURT_WIDTH, 0),     p(COURT_WIDTH, COURT_LENGTH),    color, 2)
    # Service lines (informational only)
    sline_near, sline_far = 5.485, 18.285
    cv2.line(img, p(0, sline_near),      p(COURT_WIDTH, sline_near),      color, 1)
    cv2.line(img, p(0, sline_far),       p(COURT_WIDTH, sline_far),       color, 1)
    # Center service line
    cv2.line(img, p(COURT_WIDTH/2, sline_near), p(COURT_WIDTH/2, sline_far), color, 1)
    # Net (visualized as midline)
    cv2.line(img, p(0, COURT_LENGTH/2),  p(COURT_WIDTH, COURT_LENGTH/2),  (0, 200, 200), 1)


def calibrate(video_path: str, output_path: str, frame_idx: int = 0):
    frame = grab_frame(video_path, frame_idx)
    points: list[tuple[float, float]] = []

    window = "Calibrate court corners"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    # Reasonable starting window size
    h, w = frame.shape[:2]
    cv2.resizeWindow(window, min(w, 1400), min(h, 900))

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, draw_overlay(frame, points))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord('q')):  # Esc or q
            cv2.destroyAllWindows()
            print("Calibration cancelled.")
            return False
        if key == ord('r'):
            points.clear()
        if key in (13, 10) and len(points) == 4:  # Enter
            break

    cv2.destroyAllWindows()

    H = compute_homography(np.array(points, dtype=np.float64))
    payload = {
        "video": str(Path(video_path).resolve()),
        "image_corners_px": [[float(x), float(y)] for x, y in points],
        "corner_order": ["near-left", "near-right", "far-right", "far-left"],
        "homography_pixel_to_court_m": H.tolist(),
        "court_dimensions_m": {"length": COURT_LENGTH, "width": COURT_WIDTH},
        "calibration_frame_index": frame_idx,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))
    print(f"Saved calibration to {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Pick court corners on the first frame of a video.")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", default="court_calibration.json", help="Where to save calibration JSON")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to use for calibration (default 0)")
    args = parser.parse_args()

    ok = calibrate(args.video, args.output, args.frame)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
