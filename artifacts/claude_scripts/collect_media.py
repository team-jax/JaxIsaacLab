"""Copy every video and image made in this project into artifacts/media/ under one naming format, with an index.

usage: python3 -I artifacts/claude_scripts/collect_media.py [--dry]

Name format (ASCII; fields joined by "_", words inside a field by "-"):
    {YYYY-MM-DD}_{HHMM}_{stage}_{run}_{model}_{view}_{tag}.{ext}
  date/time  when the file was written (local time, KST)
  stage      task: "v2-..." = JAX V2 (URDF_1006) tasks, "oaj-..." = the earlier one_arm_jax tasks,
             "artifacts" = files from artifacts/, "analysis" = frames grabbed while checking results
  run        training run name without its timestamp (e.g. v50-r3), "none" if not from a run
  model      checkpoint, e.g. m13075, "none" if not tied to one
  view       camera (front, side, close-front, close), "montage-front" for the 3x3 chain montages,
             "clip" for other videos, "single" for other single renders, "ref" for scripted references,
             "image"/"frame" for pictures
  tag        what else the original name said (cube offset, condition...), "none" if nothing
Files go to artifacts/media/video/{date}/ (mp4, gif) and artifacts/media/image/{date}/ (png, jpg).
Originals are copied (not moved). Skipped: logs/rsl_rl/h1_rough (Isaac Lab example from 2026-09, not this
project), artifacts/deploy_2026-11-05/video (copies of montages already collected), artifacts/media itself.
"""
import csv
import datetime
import os
import re
import shutil
import sys

ROOT = "/home/jeongwoo/isaacsim/JaxIsaacLab"
SCRATCH = "/tmp/claude-1000/-home-jeongwoo-isaacsim-JaxIsaacLab/886e2a88-a982-4003-b7f5-73532822d877/scratchpad"
OUT = os.path.join(ROOT, "artifacts", "media")
VIDEO = {".mp4", ".gif"}
IMAGE = {".png", ".jpg", ".jpeg"}
TS = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_(.*))?$")
CAM = r"(close_front|front|side|close)"
DRY = "--dry" in sys.argv


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9.+-]+", "-", s.lower())
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    return s or "none"


def stage_of(task: str) -> str:
    for pre, code in (("jax_v2_1006_", "v2-"), ("jax_v2_", "v2-"), ("one_arm_jax_minimal_", "oaj-"), ("one_arm_jax_", "oaj-")):
        if task.startswith(pre):
            return code + norm(task[len(pre):])
    return norm(task)


def model_of(s: str):
    m = re.search(r"model_(\d+)", s)
    return ("m" + m.group(1)) if m else "none"


def strip_model(s: str) -> str:
    return re.sub(r"_?model_\d+", "", s)


def from_logs(path: str):
    parts = os.path.relpath(path, os.path.join(ROOT, "logs", "rsl_rl")).split(os.sep)
    task, rest = parts[0], parts[1:]
    stem = os.path.splitext(rest[-1])[0]
    stage = stage_of(task)
    step_tag = "" if stem == "rl-video-step-0" else norm(stem)
    if len(rest) == 1:  # a reference render saved straight into the task folder
        return stage, "none", model_of(stem), "ref", step_tag or "none"
    rundir, inner = rest[0], rest[1:]
    m = TS.match(rundir)
    run = norm(m.group(1) or "run") if m else "none"
    pre_tag = "" if m else norm(rundir)
    if inner and inner[0] == "videos":
        inner = inner[1:]
    if len(inner) == 1:  # a file straight in videos/ (montage or other clip)
        mm = re.match(rf"^montage_(?:v\d+_)?{CAM}_model_(\d+)(?:_(.*))?$", stem)
        if mm:
            return stage, run, "m" + mm.group(2), "montage-" + norm(mm.group(1)), norm(mm.group(3) or "")
        view = "ref" if pre_tag else "clip"
        tag = "-".join(t for t in (pre_tag, norm(strip_model(stem))) if t and t != "none") or "none"
        return stage, run, model_of(stem), view, tag
    d = inner[0]
    md = re.match(rf"^(.*?)_{CAM}_model_(\d+)(?:_(.*))?$", d)
    if md:
        prefix, cam, model, extra = md.group(1), md.group(2), md.group(3), md.group(4) or ""
        extra = re.sub(r"^v27_", "chain_", extra)
        pre = norm(prefix)
        tags = [pre_tag] + ([] if pre in stage else [pre]) + [norm(extra) if extra else ""] + [step_tag]
        return stage, run, "m" + model, norm(cam), "-".join(t for t in tags if t and t != "none") or "none"
    view = "ref" if pre_tag else "single"
    tags = [pre_tag, norm(strip_model(d)), step_tag]
    return stage, run, model_of(d), view, "-".join(t for t in tags if t and t != "none") or "none"


def from_artifacts(path: str, kind: str):
    rel = os.path.relpath(path, os.path.join(ROOT, "artifacts"))
    dirs = [p for p in rel.split(os.sep)[:-1] if p != "frames"]
    stem = os.path.splitext(os.path.basename(rel))[0]
    view = "clip" if kind == "video" else ("frame" if "/frames/" in path else "image")
    tag = "-".join(norm(t) for t in dirs + [strip_model(stem)])
    return "artifacts", "none", model_of(rel), view, tag or "none"


def collect():
    items = []
    for base, _, files in os.walk(os.path.join(ROOT, "logs", "rsl_rl")):
        if "/h1_rough" in base:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO | IMAGE:
                p = os.path.join(base, f)
                items.append((p, "video" if ext in VIDEO else "image", from_logs(p)))
    for base, _, files in os.walk(os.path.join(ROOT, "artifacts")):
        if base.startswith(OUT) or "/deploy_2026-11-05/video" in base or "/claude_scripts" in base:
            continue
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO | IMAGE:
                p = os.path.join(base, f)
                kind = "video" if ext in VIDEO else "image"
                items.append((p, kind, from_artifacts(p, kind)))
    if os.path.isdir(SCRATCH):
        for f in sorted(os.listdir(SCRATCH)):
            if os.path.splitext(f)[1].lower() in IMAGE:
                p = os.path.join(SCRATCH, f)
                items.append((p, "image", ("analysis", "none", model_of(f), "frame", norm(os.path.splitext(f)[0]))))
    return items


def main():
    rows, used = [], set()
    for p, kind, (stage, run, model, view, tag) in sorted(collect(), key=lambda x: os.path.getmtime(x[0])):
        t = datetime.datetime.fromtimestamp(os.path.getmtime(p))
        ext = os.path.splitext(p)[1].lower()
        base = f"{t:%Y-%m-%d}_{t:%H%M}_{stage}_{run}_{model}_{view}_{tag}"
        name, k = base + ext, 2
        while name in used:
            name, k = f"{base}-{k}{ext}", k + 1
        used.add(name)
        dst = os.path.join(OUT, kind, f"{t:%Y-%m-%d}", name)
        if not DRY:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
        rows.append({"new_path": os.path.relpath(dst, OUT), "kind": kind, "made_kst": f"{t:%Y-%m-%d %H:%M}",
                     "stage": stage, "run": run, "model": model, "view": view, "tag": tag,
                     "bytes": os.path.getsize(p),
                     "original_path": p.replace(ROOT + "/", "").replace(SCRATCH, "scratchpad")})
    if not DRY:
        with open(os.path.join(OUT, "INDEX.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    nv = sum(r["kind"] == "video" for r in rows); ni = len(rows) - nv
    print(f"{'DRY ' if DRY else ''}{len(rows)} files ({nv} videos, {ni} images), {sum(r['bytes'] for r in rows) / 1e6:.1f} MB")
    for r in rows[:: max(1, len(rows) // 12)]:
        print(r["new_path"], "<-", r["original_path"])


main()
