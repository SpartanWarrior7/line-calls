"""
Chunked TrackNetV3 inference runner.
"""
import argparse
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np
import torch

DEFAULT_VIDEO = "line-training-6.mp4"
DEFAULT_CALIB = "court_calibration.json"
DEFAULT_CKPT  = "ckpt"
CHUNK_SIZE    = 13


def _setup_path():
    here = Path(__file__).resolve().parent
    repo = here.parent / "TrackNetV3"
    sys.path.insert(0, str(here))
    sys.path.insert(0, str(repo))


def _open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, orig_w, orig_h, n


def stage_bg(video_path, ckpt_dir):
    _setup_path()
    from tracknet_detector import (
        _compute_background, INPUT_HEIGHT, INPUT_WIDTH, SEQ_LEN
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cap, fps, orig_w, orig_h, n_frames = _open_video(video_path)
    cap.release()
    print(f"Video: {n_frames} frames @ {fps:.1f} fps  ({orig_w}x{orig_h})")
    print("Computing background median ...")
    bg = _compute_background(str(video_path))
    np.save(str(ckpt_dir / "bg.npy"), bg)
    n_windows = (n_frames + SEQ_LEN - 1) // SEQ_LEN
    meta = {
        "video": str(video_path),
        "fps": fps, "orig_w": orig_w, "orig_h": orig_h,
        "n_frames": n_frames, "n_windows": n_windows,
        "seq_len": SEQ_LEN,
        "next_window": 0,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Background saved -> {ckpt_dir/'bg.npy'}")
    print(f"Meta saved       -> {ckpt_dir/'meta.json'}  ({n_windows} windows total)")


def stage_infer(video_path, ckpt_dir, chunk_size, device):
    _setup_path()
    from tracknet_detector import (
        _load_model, _resize_frame, _preprocess_sequence,
        _peak_from_heatmap, _rescale_to_original,
        SEQ_LEN, CONFIDENCE_THRESHOLD
    )
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        print("No meta.json -- run --stage bg first.")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    next_w    = meta["next_window"]
    n_windows = meta["n_windows"]
    n_frames  = meta["n_frames"]
    fps       = meta["fps"]
    orig_w    = meta["orig_w"]
    orig_h    = meta["orig_h"]
    if next_w >= n_windows:
        print("All windows done. Run --stage finish.")
        return
    end_w = min(next_w + chunk_size, n_windows)
    print(f"Inference: windows {next_w}-{end_w-1} of {n_windows} ({end_w - next_w} windows)")
    bg = np.load(str(ckpt_dir / "bg.npy"))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(device)
    det_path = ckpt_dir / "detections.json"
    if det_path.exists():
        detections = json.loads(det_path.read_text())
    else:
        detections = {}
    cap, _, _, _, _ = _open_video(video_path)
    start_frame = next_w * SEQ_LEN
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    import time
    t0 = time.perf_counter()
    for w in range(next_w, end_w):
        seq = []
        for _ in range(SEQ_LEN):
            ok, frame = cap.read()
            if not ok:
                break
            seq.append(_resize_frame(frame))
        if len(seq) < SEQ_LEN:
            if not seq:
                break
            while len(seq) < SEQ_LEN:
                seq.append(seq[-1])
        tensor = _preprocess_sequence(bg, seq).to(device)
        with torch.no_grad():
            out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        for i in range(SEQ_LEN):
            frame_idx = w * SEQ_LEN + i
            if frame_idx >= n_frames:
                break
            heatmap = out[0, i].cpu().numpy()
            peak = _peak_from_heatmap(heatmap)
            if peak is not None:
                px_in, py_in, conf = peak
                x_orig, y_orig = _rescale_to_original(px_in, py_in, orig_w, orig_h)
                detections[str(frame_idx)] = {
                    "frame": frame_idx,
                    "t": frame_idx / fps,
                    "x": float(x_orig),
                    "y": float(y_orig),
                    "confidence": float(conf),
                }
    cap.release()
    elapsed = time.perf_counter() - t0
    det_path.write_text(json.dumps(detections))
    meta["next_window"] = end_w
    meta_path.write_text(json.dumps(meta, indent=2))
    done_pct = 100.0 * end_w / n_windows
    print(f"  {len(detections)} detections so far  |  {done_pct:.0f}% complete  |  {elapsed:.1f}s elapsed")
    if end_w < n_windows:
        remaining = (n_windows - end_w) * (elapsed / (end_w - next_w))
        print(f"  ~{remaining:.0f}s remaining -- run --stage infer again")
    else:
        print("  All windows complete -- run --stage finish")


def stage_finish(video_path, calib_path, ckpt_dir):
    _setup_path()
    import importlib
    process   = importlib.import_module("process")
    visualize = importlib.import_module("visualize")
    meta_path = ckpt_dir / "meta.json"
    det_path  = ckpt_dir / "detections.json"
    if not det_path.exists():
        print("No detections.json -- run inference first.")
        sys.exit(1)
    meta = json.loads(meta_path.read_text())
    raw  = json.loads(det_path.read_text())
    fps    = meta["fps"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]
    n      = meta["n_frames"]

    # -----------------------------------------------------------------------
    # Use ALL detections (TrackNet + InpaintNet fills) for trajectory splitting.
    #
    # InpaintNet only filled gaps <= max_gap=15 frames, so all remaining gaps
    # are between-rally breaks (>=21 frames in this video). Since those gaps
    # are all > MAX_TRAJECTORY_GAP_FRAMES=8, they still split trajectories.
    # This gives us full rally-length arcs that span through bounce points,
    # instead of short fragments that miss the inflection.
    # -----------------------------------------------------------------------
    n_filled = sum(1 for d in raw.values() if d.get("confidence", 1.0) < 1.0)
    n_raw    = len(raw) - n_filled
    print(f"Loaded {n_raw} TrackNet + {n_filled} InpaintNet-filled = {len(raw)} detections")

    traj_source = {}
    for k, v in raw.items():
        fi = int(k)
        traj_source[fi] = {
            "frame": fi, "t": fi / fps,
            "x": v["x"], "y": v["y"],
            "confidence": v.get("confidence", 1.0),
        }

    calib = json.loads(Path(calib_path).read_text())
    H = np.array(calib["homography_pixel_to_court_m"], dtype=np.float64)

    trajectories = process.group_into_trajectories(traj_source, fps)
    print(f"Assembled {len(trajectories)} trajectories.")

    from courtmath import pixel_to_court, classify_bounce, NET_Y, COURT_LENGTH, COURT_WIDTH
    from process import (find_bounces, Bounce, NET_ZONE_BUFFER_M, COURT_MARGIN_TOLERANCE_M)

    all_bounces = []
    rejected = 0
    for traj_idx, traj in enumerate(trajectories):
        for bi in find_bounces(traj):
            d = traj.detections[bi]
            x_m, y_m = pixel_to_court(H, d.x, d.y)
            if abs(y_m - NET_Y) < NET_ZONE_BUFFER_M:
                rejected += 1
                continue
            tol = COURT_MARGIN_TOLERANCE_M
            if x_m < -tol or x_m > COURT_WIDTH + tol or y_m < -tol or y_m > COURT_LENGTH + tol:
                rejected += 1
                continue
            result = classify_bounce(x_m, y_m)
            all_bounces.append(Bounce(
                trajectory_idx=traj_idx,
                frame=d.frame, t=d.t,
                x_px=d.x, y_px=d.y,
                x_court_m=x_m, y_court_m=y_m,
                call=result["call"], reason=result["reason"],
                margin_m=result["margin_m"],
            ))

    if rejected:
        print(f"  ({rejected} candidates rejected by plausibility filters)")

    print(f"\nDetected {len(all_bounces)} bounces:")
    for i, b in enumerate(all_bounces, 1):
        print(f"  #{i:2d}  t={b.t:6.2f}s  court=({b.x_court_m:6.2f}m, {b.y_court_m:6.2f}m)  -> {b.call}  ({b.reason})")

    from dataclasses import asdict
    payload = {
        "video": str(Path(video_path).resolve()),
        "calibration": str(Path(calib_path).resolve()),
        "fps": fps, "width": orig_w, "height": orig_h,
        "total_frames": n, "detector": "TrackNetV3",
        "trajectories": [{"detections": [asdict(d) for d in t.detections]} for t in trajectories],
        "bounces": [asdict(b) for b in all_bounces],
    }
    Path("results.json").write_text(json.dumps(payload, indent=2))
    print("\nSaved results.json")

    print("\nRendering overlay video ...")
    visualize.render_overlay(str(video_path), "results.json",
                             calibration_path=str(calib_path),
                             output_path="overlay.mp4")
    print("Rendering top-down diagram ...")
    visualize.render_topdown("results.json", "topdown.png")
    print("Done -- overlay.mp4 and topdown.png written.")


def auto_mode(video_path, calib_path, ckpt_dir, chunk_size, device):
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.exists():
        print("=== Stage: bg ===")
        stage_bg(video_path, ckpt_dir)
        return
    meta = json.loads(meta_path.read_text())
    if meta["next_window"] < meta["n_windows"]:
        print("=== Stage: infer ===")
        stage_infer(video_path, ckpt_dir, chunk_size, device)
        return
    if not Path("results.json").exists() or not Path("overlay.mp4").exists():
        print("=== Stage: finish ===")
        stage_finish(video_path, calib_path, ckpt_dir)
        return
    print("All stages complete. results.json and overlay.mp4 are up to date.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["bg", "infer", "finish", "auto"], default="auto")
    parser.add_argument("--video",       default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", default=DEFAULT_CALIB)
    parser.add_argument("--ckpt-dir",    default=DEFAULT_CKPT)
    parser.add_argument("--chunk-size",  type=int, default=CHUNK_SIZE)
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()
    video_path = Path(args.video)
    calib_path = Path(args.calibration)
    ckpt_dir   = Path(args.ckpt_dir)
    if args.stage == "bg":
        stage_bg(video_path, ckpt_dir)
    elif args.stage == "infer":
        stage_infer(video_path, ckpt_dir, args.chunk_size, args.device)
    elif args.stage == "finish":
        stage_finish(video_path, calib_path, ckpt_dir)
    elif args.stage == "auto":
        auto_mode(video_path, calib_path, ckpt_dir, args.chunk_size, args.device)
