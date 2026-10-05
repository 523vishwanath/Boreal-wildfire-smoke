import argparse, json, subprocess
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

OUT = Path("/workspace/temporal")
WEIGHTS = "/workspace/runs/boreal_seg/yolo26m_seg_960/weights/best.pt"
TRACKER = str(OUT / "botsort_smoke.yaml")

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--fps", type=float, default=4.0)
ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--frame_thr", type=float, default=0.5, help="frame rule: alert if any detection >= this")
ap.add_argument("--min_dur", type=float, default=4.0, help="persistence rule: track must last this many seconds")
ap.add_argument("--min_mean_conf", type=float, default=0.35, help="persistence rule: mean confidence")
args = ap.parse_args()

src = Path(args.video)
for d in ("inputs", "vis", "tracks"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

small = OUT / "inputs" / f"{src.stem}_960.mp4"
print(f"Shrinking {src.name} to 960 px at {args.fps:g} fps...")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"fps={args.fps},scale=960:-2",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", str(small)], check=True)

model = YOLO(WEIGHTS)
rows, writer, n = [], None, 0
tmp = OUT / "vis" / f"{src.stem}_tmp.mp4"
for i, r in enumerate(model.track(source=str(small), stream=True, tracker=TRACKER,
                                  conf=args.conf, iou=0.5, imgsz=960, verbose=False)):
    n = i + 1
    frame = r.plot(line_width=2, font_size=12)
    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    writer.write(frame)
    if r.boxes is None or r.boxes.id is None:
        continue
    polys = r.masks.xyn if r.masks is not None else [None] * len(r.boxes)
    for tid, c, b, poly in zip(r.boxes.id.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist(), polys):
        rows.append({"frame": i, "t": round(i / args.fps, 2), "track_id": tid, "conf": round(c, 4),
                     "box": [round(v, 5) for v in b],
                     "poly": [] if poly is None else np.round(poly, 5).tolist()})
if writer:
    writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(OUT / "vis" / f"{src.stem}.mp4")])
    tmp.unlink(missing_ok=True)
with open(OUT / "tracks" / f"{src.stem}.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"\n{n} frames ({n / args.fps:.1f} s), {len({r['track_id'] for r in rows})} tracks")
tracks = {}
for row in rows:
    tracks.setdefault(row["track_id"], []).append(row)
print(f"\n{'track':>5} {'start_s':>7} {'end_s':>6} {'dur_s':>6} {'cover':>6} {'max_conf':>8} {'mean_conf':>9} {'persist_alert':>13}")
persist_alerts = []
for tid, tr in sorted(tracks.items()):
    f0, f1 = tr[0]["frame"], tr[-1]["frame"]
    dur = (f1 - f0 + 1) / args.fps
    cover = len(tr) / (f1 - f0 + 1)
    confs = [x["conf"] for x in tr]
    alert = dur >= args.min_dur and np.mean(confs) >= args.min_mean_conf and cover >= 0.6
    if alert:
        persist_alerts.append((tid, tr[0]["t"]))
    print(f"{tid:5d} {tr[0]['t']:7.1f} {tr[-1]['t']:6.1f} {dur:6.1f} {cover:6.2f} {max(confs):8.2f} {np.mean(confs):9.2f} {str(alert):>13}")

frame_hits = sorted({r["t"] for r in rows if r["conf"] >= args.frame_thr})
print("\n=== Verdicts ===")
print(f"Frame rule (any detection >= {args.frame_thr}): "
      + (f"ALERT, first at {frame_hits[0]:.1f}s, {len(frame_hits)} frames over threshold" if frame_hits else "no alert"))
print(f"Persistence rule (track >= {args.min_dur:g}s, mean conf >= {args.min_mean_conf}, present >= 60% of its span): "
      + (f"ALERT on tracks {[t for t, _ in persist_alerts]}, first at {min(s for _, s in persist_alerts):.1f}s"
         if persist_alerts else "no alert"))
print(f"\nAnnotated video: {OUT / 'vis' / (src.stem + '.mp4')}")
