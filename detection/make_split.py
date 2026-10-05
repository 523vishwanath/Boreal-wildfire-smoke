#!/usr/bin/env python3
"""
Leak-free train/val/test split for the Boreal Forest Fire dataset (YOLO format).

Frames from the same drone video are near-duplicates, so we split by VIDEO,
not by frame: every frame of a video lands in the same split.
Sites with too few videos fall back to a temporal block split
(train | gap | val | gap | test) with some frames dropped between blocks.

Output (Ultralytics layout, symlinks by default):
  OUT/images/{train,val,test}/...
  OUT/labels/{train,val,test}/...
  OUT/data.yaml
  OUT/split_manifest.csv
  OUT/near_duplicates.csv   (only with --check-dupes)
"""
import argparse
import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NAME_RE = re.compile(r"^(?P<site>[A-Za-z]+)_(?P<video>.+?)_frame(?P<frame>\d+)$", re.I)
SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------
def parse_name(stem):
    """ruokolahti_DJI_0080_frame6 -> ('ruokolahti', 'ruokolahti_dji_0080', 6)"""
    m = NAME_RE.match(stem)
    if m:
        site = m["site"].lower()
        return site, f"{site}_{m['video'].lower()}", int(m["frame"])
    # Fallback: first token is the site, strip the trailing frame number for the video id
    site = stem.split("_")[0].lower()
    num = re.search(r"(\d+)$", stem)
    video = re.sub(r"[_\-]?(frame)?\d+$", "", stem, flags=re.I).lower()
    return site, video, int(num.group(1)) if num else 0


def read_label(path):
    """Return (class ids, max tokens on any line). >5 tokens means polygon labels."""
    if path is None or not path.exists():
        return [], 0
    cls, maxlen = [], 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls.append(int(float(parts[0])))
        maxlen = max(maxlen, len(parts))
    return cls, maxlen


def collect(roots):
    records, known_sites = [], set()
    for root in map(Path, roots):
        img_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower().endswith("-images"))
        if not img_dirs:
            print(f"[warn] no '*-Images' folders found in {root}")
        for img_dir in img_dirs:
            prefix = img_dir.name[: -len("-images")]
            lbl_dir = root / f"{prefix}-Labels"
            labels = {p.stem: p for p in lbl_dir.rglob("*.txt")} if lbl_dir.exists() else {}
            folder_site = None if prefix.lower() == "empty" else prefix.lower()
            if folder_site:
                known_sites.add(folder_site)

            for img in sorted(img_dir.rglob("*")):
                if img.suffix.lower() not in IMG_EXTS or img.name.startswith("._"):
                    continue
                site, video, frame = parse_name(img.stem)
                if folder_site and site != folder_site:
                    video, site = f"{folder_site}_{video}", folder_site
                lbl = labels.get(img.stem)
                classes, maxlen = read_label(lbl)
                records.append(dict(img=img, lbl=lbl, site=site, video=video, frame=frame,
                                    classes=classes, maxlen=maxlen, subset=root.name,
                                    folder=prefix, split=None))
    return records, known_sites


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------
def greedy_assign(video_sizes, ratios, rng):
    """Assign whole videos to splits, biggest first, to whichever split is furthest below target."""
    keys = list(video_sizes)
    rng.shuffle(keys)                       # random tie-break
    total = sum(video_sizes.values())
    got = {s: 0 for s in ratios}
    out = {}
    # Seed the smaller splits first with the video closest to their target size,
    # so every split gets at least one video
    for s in sorted(ratios, key=lambda s: ratios[s]):
        if len(out) >= len(keys) - 1 or s == max(ratios, key=ratios.get):
            break
        free = [k for k in keys if k not in out]
        k = min(free, key=lambda k: abs(video_sizes[k] - ratios[s] * total))
        out[k] = s
        got[s] += video_sizes[k]
    keys = sorted((k for k in keys if k not in out), key=lambda k: -video_sizes[k])
    for k in keys:
        s = max(ratios, key=lambda s: ratios[s] * total - got[s])
        out[k] = s
        got[s] += video_sizes[k]
    return out


def block_assign(recs, ratios, gap):
    """Temporal split inside one video: train | gap | val | gap | test. Gap frames are dropped."""
    recs = sorted(recs, key=lambda r: r["frame"])
    order = [s for s in SPLITS if ratios.get(s, 0) > 0]
    n, k = len(recs), len(order)
    if n - gap * (k - 1) < k:   # video too short for gaps
        gap = 0
    usable = n - gap * (k - 1)
    sizes = [round(usable * ratios[s]) for s in order[:-1]]
    sizes.append(usable - sum(sizes))
    i = 0
    for j, (s, size) in enumerate(zip(order, sizes)):
        for r in recs[i:i + size]:
            r["split"] = s
        i += size
        if j < k - 1:
            for r in recs[i:i + gap]:
                r["split"] = "dropped"
            i += gap


def apply_train_stride(records, stride):
    if stride <= 1:
        return
    by_vid = defaultdict(list)
    for r in records:
        if r["split"] == "train":
            by_vid[r["video"]].append(r)
    for recs in by_vid.values():
        for i, r in enumerate(sorted(recs, key=lambda r: r["frame"])):
            if i % stride:
                r["split"] = "dropped"


# ----------------------------------------------------------------------------
# Near-duplicate check (catches the same scene showing up in two different videos)
# ----------------------------------------------------------------------------
def _dhash(path):
    from PIL import Image
    import imagehash
    with Image.open(path) as im:
        im.draft("RGB", (512, 512))   # fast JPEG decode at low res
        return int(str(imagehash.dhash(im.convert("L"), hash_size=8)), 16)


def dupe_check(records, out, thr, workers):
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor

    kept = [r for r in records if r["split"] in SPLITS]
    print(f"\nHashing {len(kept)} images for near-duplicate check...")
    with ProcessPoolExecutor(workers) as ex:
        hashes = list(ex.map(_dhash, [str(r["img"]) for r in kept], chunksize=32))
    for r, h in zip(kept, hashes):
        r["hash"] = h

    train = [r for r in kept if r["split"] == "train"]
    if not train:
        return
    th = np.array([r["hash"] for r in train], dtype=np.uint64)
    rows = []
    for s in ("val", "test"):
        ev = [r for r in kept if r["split"] == s]
        hits = 0
        for r in ev:
            x = np.bitwise_xor(th, np.uint64(r["hash"]))
            d = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
            j = int(d.argmin())
            if d[j] <= thr:
                hits += 1
                rows.append([s, r["img"], train[j]["img"], int(d[j])])
        print(f"  {s}: {hits}/{len(ev)} images have a near-duplicate in train (dhash dist <= {thr})")
    with open(out / "near_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "closest_train_image", "hamming_dist"])
        w.writerows(rows)
    print(f"  details -> {out / 'near_duplicates.csv'}")


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def link_or_copy(src, dst, copy):
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="Subset folders, e.g. .../Boreal-Forest-Fire-Subset-A")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--min-videos", type=int, default=3, help="Sites with fewer videos use the temporal block split")
    ap.add_argument("--gap", type=int, default=10, help="Frames dropped between blocks in block mode")
    ap.add_argument("--holdout-site", default=None, help="Put a whole site in test (unseen-location test)")
    ap.add_argument("--train-stride", type=int, default=1, help="Keep every Nth train frame per video")
    ap.add_argument("--names", default="smoke", help="Comma-separated class names, index order")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check-dupes", action="store_true")
    ap.add_argument("--dupe-thr", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", nargs="*", default=[], help="Videos to treat as one, comma-separated, e.g. karkkila_DJI_0007,karkkila_DJI_0008")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tot = sum(args.ratios)
    ratios = {s: r / tot for s, r in zip(SPLITS, args.ratios) if r > 0}

    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists, pass --overwrite")
        shutil.rmtree(out)

    records, known_sites = collect(args.roots)
    print(f"Found {len(records)} images. Sites from folders: {sorted(known_sites)}")
    odd = Counter(r["site"] for r in records if r["site"] not in known_sites)
    if odd:
        print(f"[warn] some (likely Empty) images did not parse to a known site: {dict(odd)}")
        print("       they still get grouped by their own video id, but check the filenames.")

    # Drop exact duplicate stems across subsets (same frame in A and C)
    seen, deduped, dup_count = set(), [], 0
    for r in records:
        key = r["img"].stem.lower()
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(r)
    if dup_count:
        print(f"[info] dropped {dup_count} images whose filename already appeared in another subset")
    records = deduped

    # Merge videos that are really the same flight (e.g. one recording split into two files)
    for group in args.merge:
        names = [n.strip().lower() for n in group.split(",")]
        for r in records:
            base = re.sub(r"_frame\d+$", "", r["img"].stem, flags=re.I).lower()
            if base in names:
                r["video"] = "merged:" + names[0]

    # Split per site
    by_site = defaultdict(list)
    for r in records:
        by_site[r["site"]].append(r)

    holdout = args.holdout_site.lower() if args.holdout_site else None
    tv = {s: ratios[s] for s in ("train", "val") if s in ratios}
    tv = {s: v / sum(tv.values()) for s, v in tv.items()}

    site_mode = {}
    for site, recs in sorted(by_site.items()):
        if site == holdout:
            for r in recs:
                r["split"] = "test"
            site_mode[site] = "holdout"
            continue
        site_ratios = tv if holdout else ratios
        vids = defaultdict(list)
        for r in recs:
            vids[r["video"]].append(r)
        if len(vids) >= args.min_videos:
            mapping = greedy_assign({v: len(rs) for v, rs in vids.items()}, site_ratios, rng)
            for v, rs in vids.items():
                for r in rs:
                    r["split"] = mapping[v]
            site_mode[site] = f"video ({len(vids)} videos)"
        else:
            for rs in vids.values():
                block_assign(rs, site_ratios, args.gap)
            site_mode[site] = f"block ({len(vids)} videos, gap={args.gap})"

    apply_train_stride(records, args.train_stride)

    # Write dataset
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    cls_counter, poly = Counter(), False
    with open(out / "split_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "site", "video", "frame", "split", "n_objects"])
        for r in records:
            w.writerow([r["img"], r["site"], r["video"], r["frame"], r["split"], len(r["classes"])])
            if r["split"] not in SPLITS:
                continue
            name = r["img"].stem
            link_or_copy(r["img"], out / "images" / r["split"] / (name + r["img"].suffix.lower()), args.copy)
            dst_lbl = out / "labels" / r["split"] / (name + ".txt")
            if r["classes"]:
                link_or_copy(r["lbl"], dst_lbl, args.copy)
                cls_counter.update(r["classes"])
                poly |= r["maxlen"] > 5
            else:
                dst_lbl.write_text("")   # background image

    # data.yaml
    names = [n.strip() for n in args.names.split(",")]
    max_id = max(cls_counter) if cls_counter else 0
    names += [f"class_{i}" for i in range(len(names), max_id + 1)]
    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")

    # Summary
    print("\nSplit mode per site:")
    for site, mode in site_mode.items():
        print(f"  {site:12s} {mode}")
    print(f"\n{'site':12s} {'split':6s} {'videos':>6s} {'images':>7s} {'empty':>6s} {'objects':>8s}")
    for site in sorted(by_site):
        for s in SPLITS:
            rs = [r for r in by_site[site] if r["split"] == s]
            if not rs:
                continue
            print(f"{site:12s} {s:6s} {len({r['video'] for r in rs}):6d} {len(rs):7d} "
                  f"{sum(not r['classes'] for r in rs):6d} {sum(len(r['classes']) for r in rs):8d}")
    for s in SPLITS:
        print(f"TOTAL {s:6s}: {sum(r['split'] == s for r in records)} images")
    print(f"dropped (gaps/stride): {sum(r['split'] == 'dropped' for r in records)}")
    print(f"\nClass ids found: {dict(sorted(cls_counter.items()))} -> names {names}")
    print("Label format:", "POLYGONS (segmentation). Ultralytics detect converts them to boxes." if poly
          else "boxes (class cx cy w h)")

    # Sanity: no video in more than one split
    vid_splits = defaultdict(set)
    for r in records:
        if r["split"] in SPLITS:
            vid_splits[r["video"]].add(r["split"])
    shared = [v for v, ss in vid_splits.items() if len(ss) > 1]
    print(f"Videos shared across splits: {len(shared)}"
          + (" (expected only for block-mode sites)" if shared else ""))

    if args.check_dupes:
        dupe_check(records, out, args.dupe_thr, args.workers)

    print(f"\nDone. data.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
