# DIY Tennis Line-Calling System

A sub-$1,000 build that watches a tennis court with cheap phone cameras and calls
balls in or out. A neural-network ball tracker ([TrackNetV3](https://github.com/qaz812345/TrackNetV3))
finds the ball in every frame, a bounce detector finds where it landed, and a court
homography projects that landing point onto the real court to decide IN / OUT / CLOSE.

The honest accuracy target is **2–5 cm** on rally-pace shots — good enough to call
obvious balls confidently and to flag the genuinely close ones for a human instead of
faking a confident call. The full rationale, hardware plan, and architecture live in
[`tennis-line-calling-plan.md`](tennis-line-calling-plan.md). Read that first if you
want the "why"; this README is the "how do I actually run the pieces."

---

## The big picture

The project is built in phases so that a cheap, single-camera proof-of-concept either
validates the idea or kills it before any money is spent on three cameras:

1. **Phase 1 — feasibility.** One phone behind one baseline, record a rally, run
   TrackNetV3 offline, and see whether it detects the ball and locates bounces well
   enough on *your* court under *your* lighting. Everything for this lives in
   [`phase1/`](phase1/) with its own [README](phase1/README.md).
2. **Court-specific accuracy.** The stock TrackNetV3 model is trained on pro broadcast
   footage and gets weaker on a phone-on-a-fence view with amateur lighting. The fix is
   **fine-tuning the model on clips from your own court** — which means labeling where
   the ball is in a few rallies, training on them, and feeding the predictions back.
   That training loop is what most of the tooling in this top-level folder is for.
3. **Multi-camera + live calls.** Phases 2–4 in the plan: second and third cameras,
   audio-clap sync, real-time inference, and a Bluetooth speaker calling shots out loud.

This README focuses on the day-to-day loop in step 2, because that's where
`label_ball.py` and the Colab notebooks come in.

---

## Repository layout

```
Line Calls/
├── README.md                       # this file
├── tennis-line-calling-plan.md     # the full system plan, BOM, and architecture
│
├── label_ball.py                   # click-to-label tool: mark the ball frame by frame
├── build_finetune_zip.py           # package labeled clips -> finetune_data.zip
├── tracknet_finetune_colab.ipynb   # Colab: fine-tune TrackNetV3 on your clips (free GPU)
├── predict_video_colab.ipynb       # Colab: run a clip through your model -> CSV + video
│
├── finetune_data.zip               # the packaged training set (rebuilt by build_finetune_zip.py)
├── labeled_data/                   # your clips, extracted frames, and label CSVs
│   ├── Line-training-1/            #   e.g. a clip folder: video + *_ball.csv + *_frames/
│   ├── line-training-6_cropped/
│   ├── ...
│   └── completed-labeled-csv/      #   finished label CSVs kept together
│
├── TrackNetV3/                      # upstream TrackNetV3 repo (cloned); model, train, predict
└── phase1/                          # single-camera proof-of-concept pipeline
    ├── README.md                   #   step-by-step Phase 1 guide
    ├── calibrate.py                #   click 4 court corners -> homography
    ├── process.py                  #   TrackNet -> trajectories -> bounces -> in/out
    └── visualize.py                #   overlay video + top-down bounce diagram
```

---

## The training loop (the main workflow)

Each time you want the model to get better on your court, you go around this loop:

```
   record clips ──► label_ball.py ──► build_finetune_zip.py ──► finetune_data.zip
                                                                      │
                                                                      ▼
                                                     tracknet_finetune_colab.ipynb
                                                                      │
                                                                      ▼
                                                       TrackNet_finetune_best.pt
                                                                      │
                                                                      ▼
                                                     predict_video_colab.ipynb
                                                          (predict a new clip)
                                                                      │
                                                                      ▼
                                          label_ball.py --review  (verify & fix predictions)
                                                                      │
                                                          add the corrected clip back ─┐
                                                                                       │
                                                          ◄────────────────────────────┘
```

Step by step:

1. **Record and trim a rally.** Keep clips short — one rally each — because TrackNetV3
   trains on stacks of *consecutive* frames. Trim with ffmpeg, e.g.
   `ffmpeg -i in.mp4 -ss 12 -to 19 -c:v libx264 -crf 18 rally1.mp4`.
2. **Label the ball** in every frame with `label_ball.py` (see the next section). This
   produces a `<clip>_ball.csv` in the exact `Frame, Visibility, X, Y` format TrackNetV3
   expects.
3. **Package the data** by listing your `(video, csv)` pairs in `build_finetune_zip.py`
   and running it. It validates that every video frame has exactly one label row, lays
   the clips out in TrackNetV3's dataset structure, and writes `finetune_data.zip`
   (backing up any previous one to `.bak`).
4. **Fine-tune on a free GPU** with `tracknet_finetune_colab.ipynb`: open it in Google
   Colab, turn on the T4 GPU, upload `finetune_data.zip`, run the cells top to bottom,
   and download `TrackNet_finetune_best.pt`. Drop it into `TrackNetV3/exp_finetune/`.
5. **Predict on a new clip** with `predict_video_colab.ipynb`: upload your fine-tuned
   `.pt` and an `.mp4`, and it returns the clip with the predicted trajectory drawn on,
   plus a prediction CSV.
6. **Review and correct** that prediction CSV with `label_ball.py --review`. Confirmed
   and fixed frames become new ground-truth labels — add the clip back into
   `build_finetune_zip.py` and go around again. Each pass makes the model better and the
   review faster.

---

## Using `label_ball.py`

`label_ball.py` is a small OpenCV GUI for stepping through a clip frame by frame and
clicking where the ball is. It writes the CSV TrackNetV3 trains on, and it has a
**review mode** for verifying the model's own predictions.

### Setup

It only needs OpenCV and NumPy (both already in the TrackNetV3 / phase1 environment):

```bash
pip install opencv-python numpy
```

### Label a clip from scratch

```bash
python label_ball.py --video rally1.mp4
```

On first run it extracts every frame to `rally1_frames/` (so frame *i* here matches
frame *i* that TrackNetV3 will extract later), then opens the labeling window. Output
goes to `rally1_ball.csv` next to the video. Useful options:

```bash
# bigger magnifier, scaled-down display
python label_ball.py --video rally1.mp4 --scale 0.75 --loupe-zoom 6

# start in fullscreen
python label_ball.py --video rally1.mp4 --fullscreen

# label AND drop straight into a TrackNetV3 dataset layout
python label_ball.py --video rally1.mp4 --dataset-root TrackNetV3/data --split train --match 24
```

### Review and fix a model's predictions

```bash
# file pickers let you choose the model CSV (and the video) interactively
python label_ball.py --review

# or name them directly
python label_ball.py --review --video rally1.mp4 --csv rally1_ball.csv
```

In review mode the model's CSV is loaded as the starting point. Each prediction shows in
**yellow** until you sign off on it, then turns **red**. Press `c` / `Enter` to accept a
prediction unchanged, or click / nudge / mark-not-visible to fix it (any of those also
count as reviewed). Corrections are saved back over the same CSV.

### Controls

| Key / action      | What it does                                                        |
|-------------------|---------------------------------------------------------------------|
| **Left click**    | Set the ball at the cursor (Visibility = 1) and advance one frame   |
| **Right click** / `/` | Set the ball without advancing (fine adjust)                    |
| **Arrow keys**    | Nudge the current label 1 pixel                                     |
| `c` / `Enter`     | (review) Confirm this frame is correct as-is and advance            |
| `n` / `Space`     | Mark the ball NOT VISIBLE (Visibility = 0) and advance              |
| `d` / `a`         | Next / previous frame                                               |
| `w` / `e`         | Jump −10 / +10 frames                                               |
| `u`               | Jump to the first unlabeled frame (review: first unreviewed)        |
| `z`               | Clear this frame's label (undo)                                     |
| `f`               | Toggle fullscreen                                                   |
| `m`               | Open the settings menu (display mode, export, on-screen directions) |
| `s`               | Save now                                                            |
| `q` / `Esc`       | Save and quit                                                       |

The top-right **magnifier** shows a zoomed view so you can place the mark precisely; the
green crosshair is the magnifier centre and a red marker shows the exact label pixel.
**Purple dots** trace the ball's recent path. The settings menu (`m`) also has
**"Export labeled CSV,"** which writes every frame to `<clip>-labeled.csv`.

### How it saves

The CSV is `Frame, Visibility, X, Y` with **one row per frame**: `Visibility = 1` and
`X, Y` = the ball-centre pixel in the clip's original resolution when visible, or
`Visibility = 0, X = 0, Y = 0` when it's occluded or off-screen. Progress auto-saves to
a `<csv>.progress.json` sidecar, so re-running on the same clip resumes where you left
off — including which frames you've already reviewed.

> **One row per frame matters.** `build_finetune_zip.py` aborts if a clip's frame count
> and label-row count don't match, because a mismatch would corrupt training. Label the
> *whole* clip, not scattered frames.

---

## Packaging and training in more detail

**`build_finetune_zip.py`** — edit the `CLIPS` list near the top to point at each
`(video, csv)` pair you want to include, then run `python build_finetune_zip.py` from
this folder (in the OpenCV env). It validates frame/label counts, copies everything into
the `data/train/match24/{video,csv}/` layout the notebook expects, and writes
`finetune_data.zip`. The fine-tune notebook holds out the **last clip alphabetically**
as the validation split, so name clips with that in mind.

**`tracknet_finetune_colab.ipynb`** — the only file you upload to Colab is
`finetune_data.zip`. It clones TrackNetV3, downloads the pretrained weights, runs a
memory-safe data-prep step (the stock `preprocess.py` runs out of RAM on long
single-rally clips), fine-tunes for a few minutes on a T4, and lets you download
`TrackNet_finetune_best.pt`.

**`predict_video_colab.ipynb`** — a standalone notebook: upload your fine-tuned `.pt`
plus any `.mp4`, and it returns an annotated video and a prediction CSV. It runs
TrackNet-only (no InpaintNet), and builds its own background from the clip, so it works
on any resolution. Feed the prediction CSV into `label_ball.py --review` to close the
loop.

---

## Where to start

- **New to the project?** Read [`tennis-line-calling-plan.md`](tennis-line-calling-plan.md)
  for goals, budget, and architecture.
- **Want to prove it works on your court?** Do [`phase1/`](phase1/README.md) — one
  camera, one rally, offline processing.
- **Ready to make the model better?** Run the training loop above:
  `label_ball.py` → `build_finetune_zip.py` → fine-tune in Colab → predict → review →
  repeat.
