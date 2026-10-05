import csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

OUT = Path("/workspace/temporal")
W_FLOW = 480
IDENT = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
FIELDS = ["video", "track_id", "frame", "t", "conf", "area_frac", "log_area", "dlog_area",
          "cx", "cy", "vx", "vy", "base_x", "base_y", "base_vx", "base_vy", "aspect", "solidity",
          "fill", "flow_mag", "flow_coherence", "flow_entropy", "bright", "sat", "contrast_in",
          "ring_diff", "edge_in", "edge_ring", "cam_dx", "cam_dy",
          "bg_flow_mag", "flow_mag_rel", "vx_rel", "vy_rel", "base_vx_rel", "base_vy_rel"]

def video_list():
    vids = []
    for p in sorted(Path("/workspace/subsetB_960/evo").glob("*.mp4"), key=lambda p: int(p.stem.split("_")[-1])):
        vids.append((p.stem, p, OUT / "tracks" / f"{p.stem}.jsonl"))
    for p in sorted((OUT / "inputs").glob("*_960.mp4")):
        name = p.stem[:-4]
        vids.append((name, p, OUT / "tracks" / f"{name}.jsonl"))
    return [v for v in vids if v[2].exists()]

def load_frames(path):
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames

def raster(row, h, w):
    m = np.zeros((h, w), np.uint8)
    if len(row.get("poly", [])) >= 3:
        cv2.fillPoly(m, [(np.array(row["poly"]) * [w, h]).astype(np.int32)], 1)
    else:
        x1, y1, x2, y2 = (np.array(row["box"]) * [w, h, w, h]).astype(int)
        m[y1:y2, x1:x2] = 1
    return m.astype(bool)

def camera_motion(g0, g1, fg0):
    bg = np.where(fg0, 0, 255).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(g0, 400, 0.01, 8, mask=bg)
    if pts is None or len(pts) < 10:
        return IDENT
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return IDENT
    A, _ = cv2.estimateAffinePartial2D(pts[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=3)
    return A if A is not None else IDENT

def apply(A, x, y):
    return A[0, 0] * x + A[0, 1] * y + A[0, 2], A[1, 0] * x + A[1, 1] * y + A[1, 2]

def process(name, vpath, tpath, writer):
    frames = load_frames(vpath)
    h, w = frames[0].shape[:2]
    hs = round(h * W_FLOW / w)
    by_frame = defaultdict(list)
    for line in open(tpath):
        r = json.loads(line)
        by_frame[r["frame"]].append(r)
    gx, gy = np.meshgrid((np.arange(W_FLOW) + 0.5) * w / W_FLOW, (np.arange(hs) + 0.5) * h / hs)
    last, prev_masks, n = {}, None, 0
    for t, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 50, 150) > 0
        masks = {r["track_id"]: raster(r, h, w) for r in by_frame.get(t, [])}
        A, resid = IDENT, None
        if t > 0:
            g0 = cv2.cvtColor(frames[t - 1], cv2.COLOR_BGR2GRAY)
            fg0 = np.zeros((h, w), bool)
            for m in (prev_masks or {}).values():
                fg0 |= m
            fg0 = cv2.dilate(fg0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            A = camera_motion(g0, gray, fg0)
            flow = cv2.calcOpticalFlowFarneback(cv2.resize(g0, (W_FLOW, hs)), cv2.resize(gray, (W_FLOW, hs)),
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0) * (w / W_FLOW)
            px, py = apply(A, gx, gy)
            resid = np.dstack([flow[..., 0] - (px - gx), flow[..., 1] - (py - gy)])
        cam_dx, cam_dy = A[0, 2] / w, A[1, 2] / w
        for r in by_frame.get(t, []):
            tid, m = r["track_id"], masks[r["track_id"]]
            area = int(m.sum())
            if area < 20:
                continue
            ys, xs = np.nonzero(m)
            cx, cy = xs.mean(), ys.mean()
            ymax = ys.max()
            bx, by = xs[ys >= ymax - 2].mean(), float(ymax)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 0
            x1, y1, x2, y2 = r["box"]
            box_px = max((x2 - x1) * w * (y2 - y1) * h, 1)
            ring = (cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~m
            row = {"video": name, "track_id": tid, "frame": t, "t": r["t"], "conf": r["conf"],
                   "area_frac": area / (h * w), "log_area": math.log(area / (h * w)),
                   "cx": cx / w, "cy": cy / h, "base_x": bx / w, "base_y": by / h,
                   "aspect": ((y2 - y1) * h) / max((x2 - x1) * w, 1e-6),
                   "solidity": area / hull if hull > 0 else 1.0, "fill": area / box_px,
                   "bright": hsv[..., 2][m].mean() / 255, "sat": hsv[..., 1][m].mean() / 255,
                   "contrast_in": gray[m].std() / 255,
                   "ring_diff": (gray[m].mean() - gray[ring].mean()) / 255 if ring.any() else 0.0,
                   "edge_in": edges[m].mean(), "edge_ring": edges[ring].mean() if ring.any() else 0.0,
                   "cam_dx": cam_dx, "cam_dy": cam_dy}
            for k in ["dlog_area", "vx", "vy", "base_vx", "base_vy", "flow_mag", "flow_coherence", "flow_entropy",
                      "bg_flow_mag", "flow_mag_rel", "vx_rel", "vy_rel", "base_vx_rel", "base_vy_rel"]:
                row[k] = ""
            p = last.get(tid)
            if p and p["frame"] == t - 1:
                row["dlog_area"] = row["log_area"] - p["log_area"]
                ex, ey = apply(A, p["cx"], p["cy"])
                row["vx"], row["vy"] = (cx - ex) / w, (cy - ey) / w
                ex, ey = apply(A, p["bx"], p["by"])
                row["base_vx"], row["base_vy"] = (bx - ex) / w, (by - ey) / w
            if resid is not None:
                ms = cv2.resize(m.astype(np.uint8), (W_FLOW, hs), interpolation=cv2.INTER_NEAREST) > 0
                if ms.sum() >= 10:
                    v = resid[ms]
                    mag = np.linalg.norm(v, axis=1)
                    row["flow_mag"] = mag.mean() / w
                    row["flow_coherence"] = np.linalg.norm(v.mean(0)) / (mag.mean() + 1e-6)
                    hist, _ = np.histogram(np.arctan2(v[:, 1], v[:, 0]), bins=8, range=(-np.pi, np.pi), weights=mag)
                    pr = hist / (hist.sum() + 1e-9)
                    row["flow_entropy"] = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum() / np.log(8))
                    rs = cv2.resize(ring.astype(np.uint8), (W_FLOW, hs), interpolation=cv2.INTER_NEAREST) > 0
                    if rs.sum() >= 10:
                        rv = resid[rs]
                        bg = rv.mean(0)
                        bgm = np.linalg.norm(rv, axis=1).mean()
                        row["bg_flow_mag"] = bgm / w
                        row["flow_mag_rel"] = (mag.mean() - bgm) / w
                        if row["vx"] != "":
                            row["vx_rel"], row["vy_rel"] = row["vx"] - bg[0] / w, row["vy"] - bg[1] / w
                            row["base_vx_rel"] = row["base_vx"] - bg[0] / w
                            row["base_vy_rel"] = row["base_vy"] - bg[1] / w
            last[tid] = {"frame": t, "log_area": row["log_area"], "cx": cx, "cy": cy, "bx": bx, "by": by}
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
            n += 1
        prev_masks = masks
    return n

def main():
    vids = video_list()
    with open(OUT / "features.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for name, vpath, tpath in vids:
            print(f"{name:10s} {process(name, vpath, tpath, writer):5d} rows", flush=True)
    print(f"\nSaved {OUT / 'features.csv'}")

if __name__ == "__main__":
    main()
