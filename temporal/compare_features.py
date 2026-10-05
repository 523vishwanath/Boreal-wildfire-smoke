import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/workspace/temporal")
labels = {(r["clip"], int(r["track_id"])): r["label"] for r in csv.DictReader(open(OUT / "track_labels.csv"))}
tracks = defaultdict(list)
for r in csv.DictReader(open(OUT / "features.csv")):
    tracks[(r["video"], int(r["track_id"]))].append(r)

def num(rows, k):
    return np.array([float(r[k]) for r in rows if r.get(k, "") != ""])

def med(v):
    return float(np.median(v)) if len(v) else np.nan

def summarize(rows):
    s = {k: med(num(rows, k)) for k in ["conf", "flow_coherence", "flow_entropy", "flow_mag", "flow_mag_rel",
                                        "bg_flow_mag", "sat", "edge_in", "ring_diff", "solidity", "aspect"]}
    d = num(rows, "dlog_area")
    s["growth_mean"] = d.mean() if len(d) else np.nan
    s["growth_std"] = d.std() if len(d) else np.nan
    s["base_speed"] = med(np.hypot(num(rows, "base_vx"), num(rows, "base_vy")))
    s["base_speed_rel"] = med(np.hypot(num(rows, "base_vx_rel"), num(rows, "base_vy_rel")))
    s["centroid_speed"] = med(np.hypot(num(rows, "vx"), num(rows, "vy")))
    s["centroid_speed_rel"] = med(np.hypot(num(rows, "vx_rel"), num(rows, "vy_rel")))
    s["camera_speed"] = med(np.hypot(num(rows, "cam_dx"), num(rows, "cam_dy")))
    return s

def group_of(key):
    v = key[0]
    if v.startswith("evo"):
        lab = labels.get(key, "unlabeled")
        return {"smoke": "evo smoke", "not_smoke": "evo false alarm"}.get(lab)
    if v.startswith(("heinola", "karkkila", "ruokolahti")):
        return "boreal no-smoke"
    if v.upper().startswith("FP"):
        return "FP fog"
    return "stock negative"

groups = defaultdict(list)
for key, rows in tracks.items():
    g = group_of(key)
    if g and len(rows) >= 8:
        groups[g].append((key, summarize(rows)))

names = sorted(groups)
feats = list(groups[names[0]][0][1].keys())
print(f"{'feature':20s}" + "".join(f"{g:>18s}" for g in names))
print(f"{'tracks':20s}" + "".join(f"{len(groups[g]):>18d}" for g in names))
for k in feats:
    print(f"{k:20s}" + "".join(f"{np.nanmedian([s[k] for _, s in groups[g]]):>18.4f}" for g in names))

neg = [k for g in names if g not in ("evo smoke",) for k, _ in groups[g]][:4]
smoke = sorted([k for k, _ in groups.get("evo smoke", [])], key=lambda k: -len(tracks[k]))[:3]
plot_keys = ["log_area", "flow_entropy", "flow_mag_rel", "base_vy_rel"]
fig, axes = plt.subplots(len(plot_keys), 1, figsize=(10, 10), sharex=True)
for ax, k in zip(axes, plot_keys):
    for key in smoke + neg:
        rows = [r for r in tracks[key] if r.get(k, "") != ""]
        if not rows:
            continue
        t = [float(r["t"]) - float(rows[0]["t"]) for r in rows]
        ax.plot(t, [float(r[k]) for r in rows], label=f"{key[0]} #{key[1]}",
                linewidth=2.5 if key in neg else 1.2, linestyle="-" if key in neg else "--")
    ax.set_ylabel(k)
axes[0].legend(fontsize=8, ncol=2)
axes[-1].set_xlabel("seconds since track start")
fig.suptitle("Negatives (solid) vs. longest Evo smoke tracks (dashed)")
fig.tight_layout()
fig.savefig(OUT / "fog_vs_smoke.png", dpi=110)
print(f"\nSaved {OUT / 'fog_vs_smoke.png'}")
