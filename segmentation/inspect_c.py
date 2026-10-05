import re, csv
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image

C = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C")
A_MANIFEST = Path("/workspace/boreal_yolo/split_manifest.csv")
SPLITS = ["train", "valid", "test"]
IMG = {".jpg", ".jpeg", ".png"}

def vid(stem):
    v = re.sub(r"_frame\d+$", "", stem, flags=re.I).lower()
    return None if v == stem.lower() else v

def files(d):
    return sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []

def imgs(s):
    return [p for p in files(C / "images" / s) if p.suffix.lower() in IMG]

def dhash(p):
    import imagehash
    with Image.open(p) as im:
        im.draft("RGB", (512, 512))
        return int(str(imagehash.dhash(im.convert("L"))), 16)

def main():
    print("== 1. Folder contents ==")
    for top in ["images", "labels", "sam_masks", "manual_masks"]:
        for s in SPLITS:
            fs = files(C / top / s)
            if fs:
                ext = dict(Counter(p.suffix.lower() for p in fs))
                print(f"{top:13s} {s:5s} {len(fs):5d} files {ext}  e.g. {fs[0].name}")

    print("\n== 2. Do labels/masks match images (by filename)? ==")
    for s in SPLITS:
        im = {p.stem for p in imgs(s)}
        for top in ["labels", "sam_masks", "manual_masks"]:
            other = {p.stem for p in files(C / top / s)}
            if other:
                print(f"{s:5s} {top:13s} matched {len(im & other)}/{len(im)}, extra {len(other - im)}")
                if not im & other:
                    print(f"      image e.g. {sorted(im)[:2]}  {top} e.g. {sorted(other)[:2]}")
    lbl = files(C / "labels" / "train")
    if lbl:
        print(f"label file sample ({lbl[0].name}): {lbl[0].read_text()[:200]!r}")

    print("\n== 3. Videos per split (Boreal's split) ==")
    sv, unparsed = {}, []
    for s in SPLITS:
        c = Counter()
        for p in imgs(s):
            v = vid(p.stem)
            if v is None:
                unparsed.append(p.name)
            else:
                c[v] += 1
        sv[s] = c
        print(f"{s}: {sum(c.values())} images from {len(c)} videos")
        for v, n in sorted(c.items()):
            print(f"     {v:28s} {n}")
    if unparsed:
        print(f"[warn] {len(unparsed)} names don't end in _frameN, e.g. {unparsed[:3]}")

    print("\n== 4. Same video in more than one split? ==")
    for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        shared = sorted(set(sv[a]) & set(sv[b]))
        print(f"{a}/{b}: {len(shared)} shared videos {shared if shared else ''}")

    print("\n== 5. Back-to-back video numbers across splits (possible same flight) ==")
    def key(v):
        m = re.search(r"(\d+)$", v)
        return (v[: m.start()], int(m.group(1))) if m else (v, None)
    train_keys = {key(v): v for v in sv["train"]}
    found = False
    for s in ["valid", "test"]:
        for v in sv[s]:
            pre, n = key(v)
            if n is None:
                continue
            for d in (-1, 1):
                if (pre, n + d) in train_keys:
                    print(f"  {s} {v}  <->  train {train_keys[(pre, n + d)]}")
                    found = True
    if not found:
        print("  none")

    print("\n== 6. Relation to Subset A and our detection split ==")
    if A_MANIFEST.exists():
        a_frames, a_vid = {}, defaultdict(set)
        for r in csv.DictReader(open(A_MANIFEST)):
            st = Path(r["image"]).stem.lower()
            a_frames[st] = r["split"]
            if vid(st):
                a_vid[vid(st)].add(r["split"])
        for s in SPLITS:
            stems = [p.stem.lower() for p in imgs(s)]
            exact = sum(st in a_frames for st in stems)
            by_det = Counter()
            for v, n in sv[s].items():
                for sp in a_vid.get(v, {"not in A"}):
                    by_det[sp] += n
            print(f"C {s:5s}: {exact}/{len(stems)} exact frames also in A | frames by our det split: {dict(by_det)}")
    else:
        print("  detection manifest not found, skipping")

    print("\n== 7. Mask format ==")
    for top in ["sam_masks", "manual_masks"]:
        fs = [p for s in SPLITS for p in files(C / top / s)]
        for p in fs[:2]:
            m = np.array(Image.open(p))
            print(f"{top}: {p.name} shape={m.shape} dtype={m.dtype} values={np.unique(m)[:10]}")
        empty = sum(np.array(Image.open(p)).max() == 0 for p in fs)
        print(f"{top}: {empty}/{len(fs)} masks are completely empty")
    ims = imgs("train")
    if ims:
        print(f"image size e.g. {Image.open(ims[0]).size}")

    print("\n== 8. Near-duplicate check (val/test vs train) ==")
    allp = {s: imgs(s) for s in SPLITS}
    with ProcessPoolExecutor(16) as ex:
        H = {s: list(ex.map(dhash, allp[s], chunksize=16)) for s in SPLITS}
    th = np.array(H["train"], dtype=np.uint64)
    for s in ["valid", "test"]:
        hits = Counter()
        for p, h in zip(allp[s], H[s]):
            d = np.unpackbits(np.bitwise_xor(th, np.uint64(h)).view(np.uint8)).reshape(-1, 64).sum(1)
            j = int(d.argmin())
            if d[j] <= 6:
                hits[(vid(p.stem), vid(allp["train"][j].stem))] += 1
        print(f"{s}: {sum(hits.values())}/{len(allp[s])} near-duplicates of train")
        for (a, b), n in hits.most_common(8):
            print(f"     {s} {a} ~ train {b}: {n}")

if __name__ == "__main__":
    main()
