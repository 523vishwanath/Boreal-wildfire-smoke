import csv, json, subprocess
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

CLIPS = Path("/workspace/subsetB_960/evo")
OUT = Path("/workspace/temporal")
WEIGHTS = "/workspace/runs/boreal_seg/yolo26m_seg_960/weights/best.pt"
TRACKER = str(OUT / "botsort_smoke.yaml")
GT = "/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-B/Video-Ground-Truth/video-ground-truths.csv"

def load_labels():
    text = open(GT, encoding="utf-8-sig").read()
    dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t")
    return {r["Filename"].rsplit(".", 1)[0]: r["Has Smoke"]
            for r in csv.DictReader(text.splitlines(), dialect=dialect)}

def main():
    labels = load_labels()
    (OUT / "tracks").mkdir(parents=True, exist_ok=True)
    (OUT / "vis").mkdir(parents=True, exist_ok=True)
    model = YOLO(WEIGHTS)
    summary = []
    clips = sorted(CLIPS.glob("*.mp4"), key=lambda p: int(p.stem.split("_")[-1]))
    for clip in clips:
        fps = cv2.VideoCapture(str(clip)).get(cv2.CAP_PROP_FPS) or 4.0
        rows, writer, n = [], None, 0
        tmp = OUT / "vis" / f"{clip.stem}_tmp.mp4"
        results = model.track(source=str(clip), stream=True, tracker=TRACKER,
                              conf=0.05, iou=0.5, imgsz=960, verbose=False)
        for i, r in enumerate(results):
            n = i + 1
            frame = r.plot(line_width=2, font_size=12)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
            if r.boxes is None or r.boxes.id is None:
                continue
            polys = r.masks.xyn if r.masks is not None else [None] * len(r.boxes)
            for tid, c, b, poly in zip(r.boxes.id.int().tolist(), r.boxes.conf.tolist(),
                                       r.boxes.xyxyn.tolist(), polys):
                rows.append({"clip": clip.stem, "frame": i, "t": round(i / fps, 3),
                             "track_id": tid, "conf": round(c, 4),
                             "box": [round(v, 5) for v in b],
                             "poly": [] if poly is None else np.round(poly, 5).tolist()})
        if writer:
            writer.release()
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264",
                            "-crf", "23", "-pix_fmt", "yuv420p", str(OUT / "vis" / f"{clip.stem}.mp4")])
            tmp.unlink(missing_ok=True)
        with open(OUT / "tracks" / f"{clip.stem}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        tracks = {}
        for row in rows:
            tracks.setdefault(row["track_id"], []).append(row["conf"])
        lens = [len(v) for v in tracks.values()]
        strong = sum(max(v) >= 0.25 for v in tracks.values())
        rec = [clip.stem, labels.get(clip.stem, "?"), n, len(tracks), strong, len(tracks) - strong,
               int(np.median(lens)) if lens else 0, max(lens) if lens else 0]
        summary.append(rec)
        print(f"{rec[0]:8s} smoke={rec[1]:5s} frames={rec[2]:4d} tracks={rec[3]:3d} "
              f"strong={rec[4]:3d} weak={rec[5]:3d} median_len={rec[6]:3d} max_len={rec[7]:3d}", flush=True)

    with open(OUT / "track_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "has_smoke", "frames", "tracks", "strong", "weak", "median_len", "max_len"])
        w.writerows(summary)
    t = np.array([s[3:6] for s in summary])
    print(f"\nTOTAL tracks={t[:, 0].sum()} strong(max conf>=0.25)={t[:, 1].sum()} weak={t[:, 2].sum()}")

if __name__ == "__main__":
    main()
