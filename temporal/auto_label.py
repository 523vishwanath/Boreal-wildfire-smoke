import csv, json, re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
from PIL import Image
import imagehash

A_IMG = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A/Evo-Images")
A_LBL = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-A/Evo-Labels")
CLIPS = Path("/workspace/subsetB_960/evo")
TRACKS = Path("/workspace/temporal/tracks")
MANIFEST = Path("/workspace/boreal_yolo/split_manifest.csv")
OUT = Path("/workspace/temporal")
HASH_THR, PIXEL_THR = 6, 15.0     # fingerprint distance, then mean pixel difference check

def thumb(im):
    return np.asarray(im.convert("L").resize((64, 34)), dtype=np.float32)

def fingerprint(im):
    return int(str(imagehash.dhash(im.convert("L"))), 16)

def vid(stem):
    return re.sub(r"_frame\d+$", "", stem, flags=re.I).lower()

def frac_inside(det, gt):
    ix = max(0.0, min(det[2], gt[2]) - max(det[0], gt[0]))
    iy = max(0.0, min(det[3], gt[3]) - max(det[1], gt[1]))
    area = (det[2] - det[0]) * (det[3] - det[1])
    return ix * iy / area if area > 0 else 0.0

def gt_boxes(stem):
    p = A_LBL / f"{stem}.txt"
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                _, cx, cy, w, h = map(float, v)
                out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out

def main():
    split_of = {Path(r["image"]).stem: r["split"] for r in csv.DictReader(open(MANIFEST))}

    a = []
    for p in sorted(A_IMG.glob("*.jpg")):
        with Image.open(p) as im:
            im.draft("RGB", (512, 512))
            im = im.convert("RGB")
            a.append({"stem": p.stem, "hash": fingerprint(im), "thumb": thumb(im)})
    print(f"Subset A Evo frames: {len(a)}")

    cf = []
    for clip in sorted(CLIPS.glob("*.mp4"), key=lambda p: int(p.stem.split("_")[-1])):
        cap = cv2.VideoCapture(str(clip))
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cf.append((clip.stem, i, fingerprint(im), thumb(im)))
            i += 1
    print(f"Clip frames: {len(cf)}")

    H = np.array([c[2] for c in cf], dtype=np.uint64)
    matches, clip_votes = {}, defaultdict(Counter)
    for x in a:
        d = np.unpackbits(np.bitwise_xor(H, np.uint64(x["hash"])).view(np.uint8)).reshape(-1, 64).sum(1)
        for j in np.argsort(d)[:5]:
            if d[j] > HASH_THR:
                break
            if np.abs(cf[j][3] - x["thumb"]).mean() < PIXEL_THR:
                matches[(cf[j][0], cf[j][1])] = x["stem"]
                clip_votes[cf[j][0]][vid(x["stem"])] += 1
                break
    print(f"Matched {len(matches)} of {len(a)} Subset A frames to clip frames\n")

    clip_info = {}
    print(f"{'clip':8s} {'source video':18s} {'det split':10s} {'matched':>7s}")
    for clip in sorted({c[0] for c in cf}, key=lambda s: int(s.split("_")[-1])):
        if clip_votes[clip]:
            src = clip_votes[clip].most_common(1)[0][0]
            sp = Counter(split_of.get(s, "?") for (cl, _), s in matches.items() if cl == clip).most_common(1)[0][0]
            clip_info[clip] = (src, sp, sum(clip_votes[clip].values()))
        else:
            clip_info[clip] = ("none", "unmatched", 0)
        print(f"{clip:8s} {clip_info[clip][0]:18s} {clip_info[clip][1]:10s} {clip_info[clip][2]:7d}")

    rows = []
    for clip, (src, sp, _) in clip_info.items():
        p = TRACKS / f"{clip}.jsonl"
        if not p.exists():
            continue
        dets = defaultdict(list)
        votes = defaultdict(Counter)
        for line in open(p):
            r = json.loads(line)
            dets[r["track_id"]].append(r["conf"])
            key = (clip, r["frame"])
            if key in matches:
                s = max((frac_inside(r["box"], g) for g in gt_boxes(matches[key])), default=0.0)
                votes[r["track_id"]]["smoke" if s >= 0.5 else "not_smoke" if s <= 0.1 else "unsure"] += 1
        for tid, confs in dets.items():
            v = votes[tid]
            decisive = v["smoke"] + v["not_smoke"]
            if decisive == 0:
                label = "unlabeled"
            elif v["smoke"] / decisive >= 0.75:
                label = "smoke"
            elif v["not_smoke"] / decisive >= 0.75:
                label = "not_smoke"
            else:
                label = "mixed"
            rows.append([clip, tid, len(confs), round(max(confs), 3), v["smoke"], v["not_smoke"],
                         v["unsure"], label, src, sp])

    with open(OUT / "track_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "track_id", "n_dets", "max_conf", "smoke_votes", "not_smoke_votes",
                    "unsure_votes", "label", "source_video", "split"])
        w.writerows(rows)

    print(f"\nTracks: {len(rows)}")
    c = Counter((r[9], r[7]) for r in rows)
    for sp in sorted({k[0] for k in c}):
        print(f"  {sp:10s} " + "  ".join(f"{lab}={c[(sp, lab)]}" for lab in
                                         ["smoke", "not_smoke", "mixed", "unlabeled"]))
    strong = Counter(r[7] for r in rows if r[3] >= 0.25)
    weak = Counter(r[7] for r in rows if r[3] < 0.25)
    print(f"\n  strong tracks (max conf >= 0.25): {dict(strong)}")
    print(f"  weak tracks   (max conf <  0.25): {dict(weak)}")
    print(f"\nSaved {OUT / 'track_labels.csv'}")

if __name__ == "__main__":
    main()
