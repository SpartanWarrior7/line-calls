# Phase 1 — Single-camera replay proof-of-concept (TrackNetV3)

**Goal of this phase:** Find out, with the minimum possible effort, whether a phone camera on a fence on *your* court, under *your* lighting, can detect a tennis ball reliably enough to be worth building the rest of the system around.

You'll know Phase 1 succeeded if, after processing a 30-second test rally, the system:
- Detects the ball in most frames where it's in clear view.
- Finds bounces at approximately the right pixel locations.
- Projects those bounces onto the court at approximately the right (x, y) in meters.

It does **not** need to be accurate or fast yet. We're checking *feasibility*.

The detector is [TrackNetV3](https://github.com/qaz812345/TrackNetV3), a CNN purpose-built for tennis/badminton ball tracking. It's much better than naive color+motion detection in tough conditions (motion blur, occlusion, complex backgrounds) but needs PyTorch and pretrained weights.

---

## What you'll need

**Hardware**
- A phone with a video camera capable of 60+ fps at 1080p. iPhone 8 and newer (240 fps slo-mo) is ideal. Most Android flagships from the last 5 years work too.
- A way to mount the phone on the fence behind one baseline. Fence clamp + tripod head ($20–30), or zip-tied phone holder for a first test.
- A laptop. **GPU strongly recommended** for TrackNet inference (CUDA-capable Nvidia GPU, or Apple Silicon with MPS backend). CPU works but is ~10–20× slower — a 30-second test clip might take 5–10 minutes on a modern CPU.
- A partner or ball machine.

**Software**
- Python 3.10–3.12
- PyTorch (CPU or CUDA)
- OpenCV, NumPy, SciPy, tqdm
- The TrackNetV3 repo cloned, with pretrained weights downloaded

---

## The plan

| Step | What you do | How long |
|---|---|---|
| 0 | Set up TrackNetV3 (clone, weights, env) | 20 min |
| 1 | Record a 30-second test clip on the court | 15 min |
| 2 | Calibrate the camera (click 4 court corners) | 2 min |
| 3 | Process the video | 1–10 min depending on GPU/CPU |
| 4 | Visualize the output | 30 sec |
| 5 | Read the output, decide if the approach works | 5 min |

---

## Step 0 — Set up TrackNetV3

This step is the new pain point compared to the simple-detector version. Do it once.

**0a. Layout**

Have your folder structure look like this (`TrackNetV3` is a sibling of `phase1`):

```
Line Calls/
├── phase1/
│   ├── tracknet_detector.py
│   ├── process.py
│   └── ...
└── TrackNetV3/         <-- you'll clone this in 0b
```

**0b. Clone the TrackNetV3 repo**

From inside the `Line Calls/` folder (one level above `phase1/`):

```bash
git clone https://github.com/qaz812345/TrackNetV3.git
```

(If you don't have Git on Windows, install [Git for Windows](https://git-scm.com/download/win).)

**0c. Download pretrained weights**

Follow the TrackNetV3 repo's README to download the pretrained tennis model. Put it at:

```
Line Calls/TrackNetV3/ckpts/TrackNetV3_best.pt
```

(If the file is named differently, set the `TRACKNET_WEIGHTS` env var, or edit `TRACKNET_WEIGHTS` at the top of `phase1/tracknet_detector.py`.)

**0d. Install Python deps**

From inside `phase1/`:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If you have an Nvidia GPU and want CUDA acceleration, install torch *before* the other requirements, picking the wheel for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**0e. Smoke-test the wrapper**

```bash
python -c "from tracknet_detector import TrackNetBallDetector; TrackNetBallDetector()"
```

You should see:

```
Loading TrackNetV3 from model on cuda...   (or cpu/mps)
TrackNetV3 ready on cuda (input 512x288, threshold 0.5).
```

If something's wrong, you'll get a specific error pointing to the integration point that needs adjusting. The most common are:

| Error | Fix |
|---|---|
| `TrackNetV3 repo not found at ...` | Wrong location. Move the clone, or set env var `TRACKNET_REPO_PATH`. |
| `Weights not found at ...` | Wrong filename. Move/rename the .pt file, or set `TRACKNET_WEIGHTS`. |
| `Could not import 'model' from ...` | The repo has the model class somewhere else (e.g. `network.py`, `src/model.py`). Set env var `TRACKNET_MODEL_IMPORT="<module>:<ClassName>"`. |
| `TrackNetV3(in_dim=9, out_dim=3) failed` | The model class takes different kwargs. Edit `MODEL_KWARGS` near the top of `tracknet_detector.py`. |
| `Unexpected TrackNet output shape ...` | The model returns a tensor shape we didn't expect. Edit the output indexing block in `detect_video`. |

The four configuration points are clearly marked in `tracknet_detector.py` under `# CONFIGURATION`.

---

## Step 1 — Record your test clip

**Camera placement**

- Mount the phone on the fence behind one baseline.
- Height: roughly chest height to head height (~1.5–1.8 m). Higher is better.
- Aim: point down the court so the camera sees both baselines, both singles sidelines, and the net.
- Keep the camera **rock steady**. Wind moving the fence shows up as pixel jitter, which is real distance at the far line.

**Phone settings**

- Resolution: **1080p**.
- Frame rate: **240 fps** if your phone supports it, else 120, else 60.
- Format: standard MP4 or MOV.
- Turn **off** "smart HDR", "cinematic mode", etc.
- Lock focus and exposure (long-press on iPhone screen).

**What to record**

~30 seconds of practice rally. Aim some balls deliberately close to the lines.

Copy the file to your laptop and call it `test_rally.mp4`.

---

## Step 2 — Calibrate

```bash
python calibrate.py --video test_rally.mp4
```

Click the **four singles court corners** in this order:

1. **Near-left** — closest to camera, on the left.
2. **Near-right**
3. **Far-right**
4. **Far-left**

Keys: `r` to reset, `Enter` to save, `Esc`/`q` to quit.

Output: `court_calibration.json`. Re-run any time you move the camera.

---

## Step 3 — Process the video

```bash
python process.py --video test_rally.mp4 --calibration court_calibration.json
```

What happens:

1. **TrackNetV3 inference** over the entire video, frame by frame (in a 3-frame sliding window). Reports detection rate at the end.
2. **Trajectory assembly** — consecutive detections are glued into trajectories; long gaps split them.
3. **Bounce detection** per trajectory — find local maxima in image-y where vertical velocity changes sign.
4. **Court projection** — apply the homography from Step 2 to each bounce pixel.
5. **In/out classification** — accounting for ball radius (33.5 mm).

Output: `results.json` and a printed bounce summary:

```
Detected 11 bounces:
  #1  t=2.13s  court=(  4.21m,   1.08m)  -> IN   (clear by 108.0cm at near baseline)
  #2  t=3.47s  court=(  7.89m,  18.41m)  -> OUT  (4.0cm past right sideline)
  ...
```

You can pass `--device cpu` if you want to force CPU (e.g. to debug CUDA issues), or `--device cuda`/`--device mps` to be explicit.

---

## Step 4 — Visualize

```bash
python visualize.py --video test_rally.mp4 --results results.json --calibration court_calibration.json
```

Produces:
- `overlay.mp4` — your original video with court lines drawn from the homography, ball trail, bounce markers, and the in/out call text per bounce.
- `topdown.png` — top-down singles-court diagram with every bounce plotted.

---

## Step 5 — What success looks like

| Outcome | Diagnosis | Fix |
|---|---|---|
| Detection rate > 80%, bounces look right, projection matches reality | Phase 1 succeeded. Move to Phase 2. | Plan Phase 2: second camera, audio-clap sync, calibration UI. |
| Detection rate high but bounce frames off by >5 frames | Tracking jitter or smoothing too aggressive. | Lower `BOUNCE_SMOOTH_WINDOW` in `process.py`. |
| Detection rate 50–80% | TrackNet is working but losing the ball — likely motion blur on fast serves, or ball going behind player. | Try recording at higher fps; consider raising `MAX_TRAJECTORY_GAP_FRAMES`. |
| Detection rate <50% | Something's wrong. Either weights aren't loading right, or the input format mismatches the model. | Re-check the smoke-test output from Step 0e; lower `CONFIDENCE_THRESHOLD` to 0.2 to see if the model is detecting *anything*. |
| Detection looks fine, court projection way off | Calibration. | Re-run `calibrate.py`, double-check corner click order. |
| All bounces are labeled OUT | Calibration corners in wrong order. | Open `court_calibration.json`, verify `image_corners_px` order is near-left → near-right → far-right → far-left. |

---

## Files

```
phase1/
├── README.md                # this file
├── requirements.txt         # python deps (incl. torch)
├── courtmath.py             # court geometry + in/out classifier (unchanged)
├── calibrate.py             # interactive 4-corner picker (unchanged)
├── tracknet_detector.py     # TrackNetV3 wrapper — the new piece
├── process.py               # TrackNet -> trajectories -> bounces -> calls
└── visualize.py             # overlay video + top-down diagram
```

---

## Troubleshooting beyond Step 0

**Video won't open in OpenCV** — iPhone HEVC is finicky. Convert with `ffmpeg -i input.mov -c:v libx264 -crf 18 output.mp4`.

**TrackNet runs but detection rate is 0%** — usually a preprocessing mismatch. Add a `print(heatmap.max())` near the top of `_peak_from_heatmap` and check the value across frames. If the max is always tiny (<0.1) the model isn't seeing what it expects — likely wrong input normalization. Some TrackNetV3 forks expect ImageNet normalization (mean/std), not 0–1 scaling. Patch `_preprocess_triplet` accordingly.

**`CUDA out of memory`** — your GPU is too small for the resolution. Either run on CPU (`--device cpu`) or lower the input resolution by editing `INPUT_HEIGHT`/`INPUT_WIDTH` in `tracknet_detector.py`. The default 288×512 fits comfortably in 4 GB.

**Inference is very slow on CPU** — expected. Modern desktop CPU is ~5–15 fps for TrackNet inference. For Phase 1 just be patient. If it's unbearable, lower `INPUT_HEIGHT`/`INPUT_WIDTH`.

**Detection rate looks great, bounces are missed** — bounce detection works on smoothed image-y trajectories. If the smoothing window swallows the bounce, lower `BOUNCE_SMOOTH_WINDOW`. If the velocity-change threshold is rejecting real bounces (visible in the overlay video as detections but no marker), lower `MIN_BOUNCE_VELOCITY_CHANGE`.
