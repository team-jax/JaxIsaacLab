"""Animate original TensorBoard scalars; this is a 2026-11-18 visualization, not historic footage."""
from pathlib import Path

import cv2
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = Path(__file__).resolve().parents[4]
RUNS = [
    ("Original reward scale (v22, 2026-10-29)",
     "logs/rsl_rl/jax_v2_1006_raise_table_v22/2026-10-29_20-01-00_v22_table"),
    ("Reward x0.02 + no value clipping (2026-11-01)",
     "logs/rsl_rl/jax_v2_1006_raise_table_v22/2026-11-01_19-48-38_v22_rs"),
]
SERIES = []
for label, folder in RUNS:
    files = list((ROOT / folder).glob("events.out.tfevents*"))
    assert len(files) == 1, folder
    acc = event_accumulator.EventAccumulator(str(files[0]), size_guidance={"scalars": 0})
    acc.Reload()
    series = {}
    for tag in ("Train/mean_reward", "Loss/value"):
        vals = acc.Scalars(tag)
        series[tag] = (np.array([v.step for v in vals], dtype=float),
                       np.array([v.value for v in vals], dtype=float))
    SERIES.append((label, folder, series))

width, height, fps, frames = 1280, 720, 24, 240
destination = Path(__file__).parent / "case12_tensorboard_trends_new_20261118.mp4"
writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
assert writer.isOpened()
font = cv2.FONT_HERSHEY_SIMPLEX
def put(img, text, x, y, scale=0.58, color=(235, 235, 235), thick=1):
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

panels = [
    (0, "Train/mean_reward", (80, 120, 520, 220), (0, 175, 255)),
    (0, "Loss/value", (680, 120, 520, 220), (120, 205, 255)),
    (1, "Train/mean_reward", (80, 410, 520, 220), (80, 225, 130)),
    (1, "Loss/value", (680, 410, 520, 220), (175, 155, 250)),
]
for fi in range(frames):
    progress = min(1.0, fi / (frames - 25))
    img = np.full((height, width, 3), (22, 24, 29), dtype=np.uint8)
    put(img, "PPO reward-scale investigation | reconstructed from original TensorBoard logs", 40, 38, .72, (255,255,255), 2)
    put(img, "New visualization: 2026-11-18 | Reward magnitudes differ; compare trends within each run only", 40, 69, .52, (195,195,195))
    for idx, tag, (x0,y0,w,h), color in panels:
        label, folder, series = SERIES[idx]
        steps, values = series[tag]
        max_step = float(steps[-1])
        upper = float(np.nanmax(values)) * 1.05
        lower = min(0.0, float(np.nanmin(values)))
        cv2.rectangle(img, (x0,y0), (x0+w,y0+h), (50,54,62), -1)
        cv2.rectangle(img, (x0,y0), (x0+w,y0+h), (80,85,95), 1)
        title = ("Old" if idx == 0 else "Scaled") + ("  mean reward" if tag.endswith("mean_reward") else "  value loss")
        put(img, title, x0+9, y0+25, .57, color, 2)
        put(img, label, x0+9, y0+48, .37, (190,190,190))
        count = max(2, int(len(steps)*progress))
        pts = []
        for step, val in zip(steps[:count], values[:count]):
            xp = int(x0+20+(w-40)*step/max_step)
            yp = int(y0+h-18-(h-83)*(val-lower)/max(1e-9,upper-lower))
            pts.append((xp,yp))
        cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        put(img, f"iteration {int(steps[count-1])}/{int(max_step)}  value {values[count-1]:.3g}", x0+12, y0+h-7, .48, (225,225,225))
    put(img, "Sources: 2026-10-29 v22_table and 2026-11-01 v22_rs event files", 45, 685, .5, (205,205,205))
    writer.write(img)
writer.release()
cap = cv2.VideoCapture(str(destination)); assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
cap.release()
print(destination)
