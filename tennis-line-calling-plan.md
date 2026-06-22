# DIY Tennis Line-Calling System — Build Plan

**Target:** Sub-$1k budget, outdoor court with fence/post mounting, live in/out calls + on-demand replay review, "tournament-credible at amateur level" (~1 cm accuracy on close calls).

---

## 1. Reality check before we start

Pro systems (Hawk-Eye, FoxTenn) hit ~3.6 mm accuracy with 10+ high-speed cameras running 340 fps, hardware-synchronized, and a calibrated 3D triangulation pipeline. They cost $50k–$100k+ per court.

At sub-$1k, the honest expectation is:

- **2–5 cm typical accuracy** on rally-pace groundstrokes (ball at ~20 m/s, bouncing).
- **5–10 cm worst case** on fast first serves (ball at 40–55 m/s, lots of motion blur).
- **~1 cm best case** on slower bounces with good lighting and the camera looking nearly straight down at the line.

This is *good enough* to:
- Confidently call balls clearly in or clearly out.
- Resolve ~70–80% of close calls that get challenged.
- Catch the truly close ones (within a centimeter of the line) only sometimes — and the system should *say so* rather than fake a confident call.

The architecture below leans into that: **live calls are advisory, and the system surfaces a confidence score so a human is the final arbiter on margin-of-error bounces.** That's both honest and the only design that actually holds up at this budget.

---

## 2. System architecture (high level)

```
                      ┌────────────────────────────────┐
                      │   Compute box (laptop or       │
                      │   Jetson Orin Nano on cart)    │
                      │   - Ball detection (TrackNet)  │
                      │   - Bounce detection           │
                      │   - Court homography           │
                      │   - In/Out decision + replay   │
                      └──────────────┬─────────────────┘
                                     │ USB / RTSP / Wi-Fi
            ┌────────────────────────┼─────────────────────────┐
            │                        │                         │
       ┌────▼────┐              ┌────▼────┐               ┌────▼────┐
       │ Cam 1   │              │ Cam 2   │               │ Cam 3   │
       │ Baseline│              │ Baseline│               │ Service │
       │ (near)  │              │ (far)   │               │ /side   │
       └─────────┘              └─────────┘               └─────────┘

  Outputs:
   - Speaker / Bluetooth puck on net post → live "OUT" / "IN" / "close"
   - Phone or tablet web UI → replay last point with bounce overlay
```

Two design choices worth flagging up front:

1. **Per-line camera coverage beats trying to see everything.** A single camera trying to call every line on the court will be wrong a lot. Instead, position each camera to look *along* a baseline or sideline — that angle minimizes parallax error at the line you actually care about.
2. **Rolling buffer + bounce-triggered replay**, not continuous full-resolution recording. You only need ~3–5 seconds of footage per point, captured around detected bounces.

---

## 3. Hardware plan (BOM at ~$950)

| Item | Choice | Qty | Unit | Total | Notes |
|---|---|---|---|---|---|
| Cameras | Used iPhone SE 2nd-gen or iPhone XR | 3 | $120 | $360 | 240 fps slo-mo at 1080p, decent rolling-shutter behavior, you probably have one already. Alternative: GoPro Hero 8/9 used at $150–200. |
| Mounts | Magnetic + fence clamp mounts | 3 | $25 | $75 | Aluminum fence clamps with 1/4"-20 thread; rubberized to dampen vibration. |
| Power | USB-C battery banks (20k mAh) | 3 | $30 | $90 | 4–5 hour runtime each. Removes need to run AC to fence posts. |
| Compute | Used laptop w/ Nvidia GTX 1660+ or M1 MacBook Air | 1 | $300–400 | $350 | Need a GPU or Apple Silicon for real-time ball detection. |
| Networking | Travel Wi-Fi router (GL.iNet Slate AX or similar) | 1 | $80 | $80 | Local-only network, no internet required courtside. Phones stream RTSP. |
| Audio | Bluetooth speaker (loud, IPX5+) | 1 | $40 | $40 | Calls "out" / "long" / "wide" so you don't need to look at a screen. |
| Calibration kit | 4× brightly colored cones + tape measure | 1 | $25 | $25 | For court reference points. |
| **Total** | | | | **~$1,020** | Trim by reusing existing phones / laptop. |

**Why three cameras, not two or four:**
- One camera can't reliably call all four sidelines plus baselines with the parallax discipline we need.
- Two cameras at opposite baselines cover both baselines and both sidelines along the long axis well; that's the MVP.
- A third camera covers the service line, which has the most disputed bounces in serves.
- Four would be ideal (one per quadrant) but pushes over budget and adds calibration/sync pain.

**Apps for the iPhones:** [Live:Air Solo](https://apps.apple.com/us/app/live-air-solo/id1296132971), [Larix Broadcaster](https://softvelum.com/larix/), or [NDI HX Camera](https://www.ndi.tv/tools/) — all stream RTSP/NDI to the laptop over Wi-Fi at 60 fps. For 240 fps you record locally to the phone and pull files after each point (acceptable for replay mode, not live).

**The 60 fps vs 240 fps fork in the road:** This is the single biggest tradeoff in the build. 60 fps streaming gives you live calls but 25–40 cm of ball travel between frames means more bounce-localization error. 240 fps recorded locally gives you the accuracy but adds ~5–10 second latency for the replay path. **Recommendation: run both. 60 fps live stream → advisory call; the same camera also records 240 fps to local storage, and the system pulls that for challenge review.**

---

## 4. Camera placement

```
                   ┌─────── 23.77 m ───────┐
                   │                       │
        Cam 1  ▼   │       NET (1.07m)     │   ▼  Cam 2
                   │   ─────────────────   │
                   │                       │
                   │      Service line     │
                   │   ─────────────────   │
                   │                       │
                   │     [singles court]   │
                   │   8.23 m × 23.77 m    │
                   │                       │
                   │      Service line     │
                   │   ─────────────────   │
                   │                       │
                   │       NET (1.07m)     │
                   │   ─────────────────   │
                   │                       │
        Cam 3 ▼     (on side, at service line height, ~5m back)
```

Each baseline camera mounts on the fence behind the *opposite* baseline at ~2.5–3 m height, looking *down the court along the centerline*. This gives a near-grazing angle on the far baseline — exactly the angle that minimizes parallax error for in/out calls on that line. It also sees both sidelines clearly.

The service-line camera mounts on the side fence at the net post extension, at ~2 m height. Its job is the service box lines, where the steepest viewing angles otherwise live.

Mounting practicals:
- **Vibration is the enemy.** Fence movement in wind shows up as 1–2 px jitter, which is 1–2 cm of apparent ball movement at distance. Use rubberized clamps and pick the heaviest section of fence.
- **Sun direction.** Mount so the camera never faces into the sun during your usual play hours. Re-evaluate seasonally.
- **Don't move the cameras between sessions** if you can help it. Recalibration takes 5 minutes but is the most common source of error.

---

## 5. Software stack

The pipeline, per camera:

1. **Frame ingest** — RTSP stream at 60 fps (live) or local file pull (240 fps replay).
2. **Court calibration** — one-time per session: detect the 4 corners of the court + service-line intersections. Compute a homography mapping pixel coordinates → court coordinates in meters.
3. **Ball detection** — [TrackNetV3](https://github.com/qaz812345/TrackNetV3) is an open-source CNN purpose-built for small/fast ball tracking in racquet sports. Trained weights are available. Output: ball (x, y) in pixel coords per frame, with confidence.
4. **Trajectory assembly** — smooth raw detections, reject outliers, fit a piecewise parabolic trajectory.
5. **Bounce detection** — find local minima in vertical position; bounce frame = the frame where dy/dt reverses sign. Sub-frame interpolation gives the actual bounce moment.
6. **Coordinate projection** — apply homography to the bounce (x, y) → court coordinates.
7. **Multi-camera fusion** — for each bounce, take the camera with the best viewing angle for that part of the court (smallest expected error). Optionally average between two cameras when both have good angles.
8. **In/out decision** — compare bounce position to the line, accounting for ball radius (33 mm). A ball is "in" if any part touches the line. Output: `IN` / `OUT` / `CLOSE — review recommended`, plus an error bar.
9. **Output** — push call to Bluetooth speaker; push replay clip + bounce overlay to web UI.

Concrete dependencies:
- Python 3.11, PyTorch, OpenCV, FFmpeg
- TrackNetV3 weights (Apache 2.0)
- A Flask or FastAPI server for the replay UI
- `mediapipe` or YOLOv8 as a fallback detector when TrackNet's confidence is low

**Sync between cameras** is the subtle hard problem. Options:
- **NTP-only:** ~50 ms drift between phones. Bad — 50 ms at 40 m/s is 2 m of ball travel.
- **Audio clap calibration:** clap once at start of session, find the clap in each audio track, align timestamps. Gets you to ~5–10 ms. **This is the recommended approach.**
- **Visual flash:** flash a light visible to all cameras at start of point. Same accuracy as audio, slightly more setup.
- **PTP / hardware trigger:** ideal but requires industrial cameras, out of budget.

5–10 ms residual sync error = 20–40 cm of ball position uncertainty in the air, but at the *bounce* the ball is at a known position regardless of sync, so the impact on the in/out call itself is small — sync mostly matters for triangulating mid-flight position, which we don't strictly need for this design.

---

## 6. Calibration procedure (do this every session)

1. Place a brightly colored cone at each of the four singles court corners.
2. From each camera, capture a still.
3. In the calibration UI, click each cone in the image. The software knows the real-world coordinates of those points (court is a fixed size).
4. Compute the homography for each camera. Sanity-check by clicking the center service line intersection in the image and confirming it maps to (0, 0) in court coordinates within 1 cm.
5. Save calibration. Re-run whenever a camera is touched.

This takes ~5 minutes total and is the single largest accuracy lever you have.

---

## 7. Build in phases, not all at once

| Phase | Goal | Time | What you learn |
|---|---|---|---|
| 1. Single-camera replay-only | One phone, recording at 240 fps from one baseline. Manual trigger. Pull file, run detector + bounce detection offline. Render overlay. | Weekend | Whether TrackNetV3 detects your ball reliably under your lighting; whether the homography is accurate enough. This phase tells you if the rest is worth building. |
| 2. Add second camera + calibration UI | Two-camera baseline coverage, automatic calibration, replay UI on phone. Still post-point only. | 2–3 weekends | Multi-camera fusion math; sync via audio clap. |
| 3. Live mode | RTSP streaming, real-time inference loop, Bluetooth speaker output. | 2 weekends | Latency budget. Real-world is the bottleneck — getting end-to-end latency under 2 seconds is the test. |
| 4. Add service-line camera + tune | Third camera for service boxes. Tune confidence thresholds against ground-truth data you collect by manually marking bounce positions on 100+ points. | 1 weekend + ongoing | Where the system is weak; where it's strong. |

**Don't skip Phase 1.** A single-camera proof-of-concept either validates the approach or surfaces a deal-breaker (lighting too inconsistent, your court has a weird color, fence vibration is worse than expected) before you've sunk money into three cameras.

---

## 8. Accuracy: how to actually measure it

A line-calling system you can't validate is just a confident guesser. Build this into the project from day one:

1. **Ground-truth dataset.** Tape ~20 small markers along the singles sidelines and baseline at known positions (every 50 cm). For one practice session, hit balls aimed at each marker. Record everything. You now have ~100 bounces with known true positions to within ~2 cm.
2. **Bench against ground truth.** Run the system over the recorded session and compute error per bounce. Report mean error, p95 error, and "fraction of calls where confidence interval excludes the true result." This is your accuracy metric.
3. **Re-run after every major change** to know if you actually improved things.

For amateur-tournament credibility, the bar I'd hold the system to:
- p95 error < 3 cm on rally-pace bounces.
- p95 error < 6 cm on serves.
- System abstains ("close — manual review") rather than calling in/out when its confidence interval crosses the line.

If those numbers hold up across a few sessions in different lighting, you have something genuinely useful.

---

## 9. Risks & mitigations

- **Lighting / shadows on outdoor courts.** Mid-afternoon harsh shadow lines can fool ball detection. Mitigation: train/fine-tune TrackNet on a small dataset from *your* court. A few hundred labeled frames go a long way.
- **Ball color variation.** Old yellow balls and bright new ones look different to a detector. Mitigation: same — court-specific fine-tuning.
- **Wind moving the fence.** See mounting notes. If it's bad, you'll need to either re-mount or implement frame-by-frame court recalibration (track court line positions every N frames).
- **Phone overheating in sun.** iPhones throttle hard above ~40°C. Mitigation: shade the phones; consider Pi HQ Camera setups as a v2 upgrade.
- **Sync drift over a long match.** Audio clap calibration drifts over hours. Re-clap at every set change.
- **"It got the close one wrong" trust collapse.** This kills adoption fast. Mitigation: the abstain-on-close-calls design above. A system that says "I'm not sure" is more trusted than one that's wrong with confidence.

---

## 10. Things to decide before building

A few open questions whose answers shape the build:

- **Court surface.** Hard court vs. clay vs. grass matters: clay leaves a mark and changes the whole problem (you can review the mark visually). If this is clay, the system design simplifies dramatically — you mainly need a high-res overhead shot of the bounce mark, not real-time trajectory tracking.
- **Singles or doubles.** Doubles sidelines extend the coverage problem; you'd want one more camera or accept lower accuracy on the doubles alleys.
- **Who's running it.** If it's just you, the laptop sits courtside on a cart. If you want others to use it without you, you'll need a more polished one-button UI than a hobbyist build typically gets to.

If you want, I can take this to the next level of detail on whichever piece is most useful next — e.g., a concrete week-by-week Phase 1 implementation guide, a parts list with exact links, or a deeper dive on the TrackNet + bounce-detection code.
