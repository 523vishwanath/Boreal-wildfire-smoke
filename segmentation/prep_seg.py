import csv, re, shutil
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

C = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C")
MANIFEST = Path("/workspace/boreal_yolo/split_manifest.csv")
OUT = Path("/root/boreal_seg_960")   # local disk (fast). Rerun this script if the pod restarts.
LONG = 960
MIN_AREA = 0.0005                    # ignore mask blobs smaller than 0.05% of the image
FOLDERS = ("train", "val", "test", "test_manual", "boreal_test")

def vid(stem):
    return re.sub(r"_frame\d+$", "", stem, flags=re.I).lower()

def mask_to_polys(m):
    h, w = m.shape
    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA * h * w:
            continue
        c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = np.clip(c / [w, h], 0, 1)
        lines.append("0 " + " ".join(f"{x:.5f} {y:.5f}" for x, y in pts))
    return lines

def job(a):
    img, mask, folder, stem = a
    im = cv2.imread(str(img))
    h, w = im.shape[:2]
    s = LONG / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / "images" / folder / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = mask_to_polys(cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)) if mask else []
    (OUT / "labels" / folder / f"{stem}.txt").write_text("".join(l + "\n" for l in lines))
    return folder, len(lines)

def main():
    # Detection split: frame -> split, video -> split as a fallback
    det, det_vid, empties = {}, {}, []
    for r in csv.DictReader(open(MANIFEST)):
        p = Path(r["image"])
        det[p.stem.lower()] = r["split"]
        det_vid[vid(p.stem)] = r["split"]
        if r["n_objects"] == "0" and r["split"] in ("train", "val", "test"):
            empties.append((p, r["split"]))

    jobs, skipped, c_stems = [], Counter(), set()
    for sub in ("train", "valid", "test"):          # Boreal's folder names (ignored for splitting)
        for img in sorted((C / "images" / sub).glob("*.jpg")):
            st = img.stem
            c_stems.add(st.lower())
            manual = C / "manual_masks" / sub / f"{st}.png"
            sam = C / "sam_masks" / sub / f"{st}.png"
            mask = manual if manual.exists() else (sam if sam.exists() else None)
            if mask is None:
                skipped["no mask"] += 1
                continue
            split = det.get(st.lower()) or det_vid.get(vid(st))
            if split is None:
                skipped["video not in detection split"] += 1
                continue
            jobs.append((img, mask, split, st))
            if manual.exists():
                jobs.append((img, manual, "boreal_test", st))
                if split == "test":
                    jobs.append((img, manual, "test_manual", st))

    n_neg = 0
    for p, split in empties:              # smoke-free frames from Subset A as negatives
        if p.stem.lower() not in c_stems:
            jobs.append((p, None, split, p.stem))
            n_neg += 1

    if OUT.exists():
        shutil.rmtree(OUT)
    for f in FOLDERS:
        (OUT / "images" / f).mkdir(parents=True)
        (OUT / "labels" / f).mkdir(parents=True)

    print(f"Processing {len(jobs)} images...")
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(job, jobs, chunksize=8))

    stats = defaultdict(lambda: [0, 0, 0])
    for folder, n in res:
        stats[folder][0] += 1
        stats[folder][1] += n == 0
        stats[folder][2] += n
    print(f"\n{'folder':12s} {'images':>7s} {'empty':>6s} {'polygons':>9s}")
    for f in FOLDERS:
        i, e, n = stats[f]
        print(f"{f:12s} {i:7d} {e:6d} {n:9d}")
    print(f"\nSkipped: {dict(skipped)} | added {n_neg} smoke-free frames from Subset A")

    ious = []
    for m in sorted((C / "manual_masks" / "test").glob("*.png")):
        s = C / "sam_masks" / "test" / m.name
        if s.exists():
            a, b = cv2.imread(str(m), 0) > 127, cv2.imread(str(s), 0) > 127
            ious.append((a & b).sum() / max((a | b).sum(), 1))
    if ious:
        print(f"SAM vs hand-drawn masks on {len(ious)} frames: mean IoU {np.mean(ious):.3f}, "
              f"median {np.median(ious):.3f}, worst {np.min(ious):.3f}")

    for name, test in [("data.yaml", "test"), ("data_manual.yaml", "test_manual"), ("data_boreal.yaml", "boreal_test")]:
        (OUT / name).write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                                f"test: images/{test}\nnames:\n  0: smoke\n")
    print(f"\nDone -> {OUT}")

if __name__ == "__main__":
    main()
