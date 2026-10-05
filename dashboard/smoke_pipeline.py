import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

WEIGHTS = os.environ.get("SMOKE_WEIGHTS", str(Path(__file__).resolve().parent.parent / "weights" / "yolo26m_seg_smoke.pt"))
TRACKER = os.environ.get("SMOKE_TRACKER", str(Path(__file__).resolve().parent.parent / "temporal" / "botsort_smoke.yaml"))
FPS = 4.0
RED, ORANGE, GREEN = (40, 40, 230), (0, 165, 255), (60, 170, 60)   # BGR


def timeline_figure(tracks, threshold, duration):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for tid, tr in sorted(tracks.items()):
        if len(tr["times"]) < 2:
            continue
        ax.plot(tr["times"], tr["scores"], label=f"track {tid}", linewidth=1.6)
        if tr["alert"]:
            ax.scatter([tr["alert"]["time_s"]], [tr["alert"]["score"]], color="red", zorder=5, s=40)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label="alert threshold")
    ax.set_xlim(0, max(duration, 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("smoke score")
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


class SmokeDetector:
    def __init__(self, weights=WEIGHTS, tracker=TRACKER):
        self.model = YOLO(weights)
        self.tracker = tracker

    @staticmethod
    def shrink(src, dst, max_seconds):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-t", str(max_seconds), "-i", str(src),
                        "-vf", f"fps={FPS},scale=960:-2", "-c:v", "libx264", "-crf", "18",
                        "-preset", "fast", "-an", str(dst)], check=True)

    def run(self, video, out_dir, threshold=0.6, min_dur=3.0, window=8, min_cover=0.6,
            max_seconds=180, progress=None):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        say = progress or (lambda f, d: None)
        say(0.0, "Preparing video")
        small = out_dir / "input_960.mp4"
        self.shrink(video, small, max_seconds)
        total = int(cv2.VideoCapture(str(small)).get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        tracks, alerts, writer, n = {}, [], None, 0
        raw = out_dir / "annotated_raw.mp4"
        results = self.model.track(source=str(small), stream=True, tracker=self.tracker,
                                   conf=0.05, iou=0.5, imgsz=960, verbose=False)
        for i, r in enumerate(results):
            n = i + 1
            t = i / FPS
            frame = r.orig_img.copy()
            dets = []
            if r.boxes is not None and r.boxes.id is not None:
                polys = r.masks.xy if r.masks is not None else [None] * len(r.boxes)
                for tid, c, b, poly in zip(r.boxes.id.int().tolist(), r.boxes.conf.tolist(),
                                           r.boxes.xyxy.tolist(), polys):
                    tr = tracks.setdefault(tid, {"frames": [], "confs": [], "scores": [], "times": [], "alert": None})
                    tr["frames"].append(i)
                    tr["confs"].append(c)
                    tr["times"].append(t)
                    score = float(np.mean(tr["confs"][-window:]))
                    tr["scores"].append(score)
                    span = i - tr["frames"][0] + 1
                    age, cover = span / FPS, len(tr["frames"]) / span
                    if tr["alert"] is None and age >= min_dur and score >= threshold and cover >= min_cover:
                        x1, y1, x2, y2 = map(int, b)
                        pad = int(0.1 * max(x2 - x1, y2 - y1))
                        crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                        tr["alert"] = {"time_s": round(t, 2), "track": tid, "score": round(score, 3),
                                       "track_age_s": round(age, 2)}
                        alerts.append((tr["alert"], cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    dets.append((tid, b, poly, score, tr["alert"] is not None))

            overlay = frame.copy()
            for tid, b, poly, score, alerted in dets:
                if poly is not None and len(poly) >= 3:
                    cv2.fillPoly(overlay, [poly.astype(np.int32)], RED if alerted else ORANGE)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            for tid, b, poly, score, alerted in dets:
                color = RED if alerted else ORANGE
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"#{tid} {score:.2f}" + (" ALERT" if alerted else "")
                cv2.putText(frame, label, (x1 + 3, max(y1 - 6, 42)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            active = any(d[4] for d in dets)
            banner = RED if active else (ORANGE if alerts else GREEN)
            text = "SMOKE ALERT" if active else ("Alert raised earlier" if alerts else "Monitoring")
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), banner, -1)
            cv2.putText(frame, f"{text}   t={t:5.1f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
            writer.write(frame)
            say(0.05 + 0.9 * n / total, "Detecting and tracking smoke")

        if writer is None:
            raise RuntimeError("Could not read any frames from this video.")
        writer.release()
        say(0.97, "Encoding result")
        final = out_dir / "annotated.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(final)], check=True)
        raw.unlink(missing_ok=True)

        rows = [{"track": tid, "start_s": round(tr["times"][0], 2), "end_s": round(tr["times"][-1], 2),
                 "duration_s": round((tr["frames"][-1] - tr["frames"][0] + 1) / FPS, 2),
                 "mean_conf": round(float(np.mean(tr["confs"])), 3),
                 "max_score": round(float(max(tr["scores"])), 3),
                 "alert": "yes" if tr["alert"] else "no"} for tid, tr in sorted(tracks.items())]
        tracks_df = pd.DataFrame(rows, columns=["track", "start_s", "end_s", "duration_s", "mean_conf", "max_score", "alert"])
        alerts_df = pd.DataFrame([a for a, _ in alerts], columns=["time_s", "track", "score", "track_age_s"])
        gallery = [(img, f"Track {a['track']} at {a['time_s']} s (score {a['score']})") for a, img in alerts]

        if alerts:
            first = alerts[0][0]
            verdict = ("<div style='padding:14px;border-radius:8px;background:#fde8e8;color:#9b1c1c'>"
                       "<b style='font-size:20px'>SMOKE ALERT</b><br>"
                       f"First alert at {first['time_s']} s on track {first['track']} (score {first['score']}). "
                       f"{len(alerts)} alert(s) in total.</div>")
        else:
            verdict = ("<div style='padding:14px;border-radius:8px;background:#e7f6ec;color:#1e6b35'>"
                       "<b style='font-size:20px'>No smoke alert</b><br>"
                       f"{len(tracks)} region(s) tracked, none met the alert rule.</div>")

        return {"video": str(final), "verdict": verdict, "alerts_df": alerts_df, "gallery": gallery,
                "figure": timeline_figure(tracks, threshold, n / FPS), "tracks_df": tracks_df}
