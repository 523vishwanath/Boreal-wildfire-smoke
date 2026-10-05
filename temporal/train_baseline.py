import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

OUT = Path("/workspace/temporal")
EXCLUDE = set()   # e.g. {("heinola_62", 3)} if a "no smoke" track turns out to be real smoke

lab = {(r["clip"], int(r["track_id"])): r for r in csv.DictReader(open(OUT / "track_labels.csv"))}
tracks = defaultdict(list)
for r in csv.DictReader(open(OUT / "features.csv")):
    tracks[(r["video"], int(r["track_id"]))].append(r)

def vals(rows, k):
    return np.array([float(r[k]) for r in rows if r.get(k, "") != ""])

def med(v):
    return float(np.median(v)) if len(v) else np.nan

def sd(v):
    return float(np.std(v)) if len(v) else np.nan

def summarize(rows):
    conf, rd = vals(rows, "conf"), vals(rows, "ring_diff")
    return {
        "conf_mean": conf.mean(), "conf_std": conf.std(),
        "coherence": med(vals(rows, "flow_coherence")), "entropy": med(vals(rows, "flow_entropy")),
        "ring_diff": med(rd), "ring_diff_std": sd(rd),
        "edge_in": med(vals(rows, "edge_in")), "sat": med(vals(rows, "sat")),
        "solidity": med(vals(rows, "solidity")), "aspect": med(vals(rows, "aspect")),
        "growth_std": sd(vals(rows, "dlog_area")),
        "base_speed_rel": med(np.hypot(vals(rows, "base_vx_rel"), vals(rows, "base_vy_rel"))),
        "centroid_speed_rel": med(np.hypot(vals(rows, "vx_rel"), vals(rows, "vy_rel"))),
        "camera_speed": med(np.hypot(vals(rows, "cam_dx"), vals(rows, "cam_dy"))),
    }

data, hold = [], []
for key, rows in tracks.items():
    if len(rows) < 8 or key in EXCLUDE:
        continue
    v = key[0]
    if v.startswith("evo"):
        L = lab.get(key)
        if not L or L["label"] not in ("smoke", "not_smoke"):
            continue
        data.append((key, summarize(rows), 1 if L["label"] == "smoke" else 0, L["source_video"]))
    elif v.startswith(("heinola", "karkkila", "ruokolahti")):
        data.append((key, summarize(rows), 0, v))
    elif v.upper().startswith("FP"):
        hold.append((key, summarize(rows)))

y = np.array([d[2] for d in data])
groups = np.array([d[3] for d in data])
print(f"Training tracks: {len(data)} ({y.sum()} smoke, {(y == 0).sum()} false alarm), "
      f"{len(set(groups))} video groups | held out: {[k for k, _ in hold]}\n")

SETS = {
    "1. confidence only": ["conf_mean", "conf_std"],
    "2. behaviour, no drift": ["conf_mean", "conf_std", "coherence", "entropy", "ring_diff", "ring_diff_std",
                               "edge_in", "sat", "solidity", "aspect", "growth_std"],
    "3. behaviour + drift": ["conf_mean", "conf_std", "coherence", "entropy", "ring_diff", "ring_diff_std",
                             "edge_in", "sat", "solidity", "aspect", "growth_std",
                             "base_speed_rel", "centroid_speed_rel", "camera_speed"],
}

def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))

for name, feats in SETS.items():
    X = np.array([[d[1][f] for f in feats] for d in data])
    oof = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = 0.5
            continue
        m = model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    caught = ((oof < 0.5) & (y == 0)).sum()
    kept = ((oof >= 0.5) & (y == 1)).sum()
    m = model().fit(X, y)
    Xh = np.array([[s[f] for f in feats] for _, s in hold]) if hold else np.empty((0, len(feats)))
    ph = m.predict_proba(Xh)[:, 1] if len(Xh) else []
    print(f"=== {name} ===")
    print(f"  leave-one-video-out AUC: {auc:.3f}")
    print(f"  false alarms rejected: {caught}/{(y == 0).sum()}   smoke kept: {kept}/{y.sum()}   (threshold 0.5)")
    print("  held-out FP34 tracks, smoke probability: "
          + ", ".join(f"#{k[1]}={p:.2f}" for (k, _), p in zip(hold, ph)))
    negs = [(d[0], p) for d, p in zip(data, oof) if d[2] == 0]
    print("  false-alarm tracks (out-of-fold prob): "
          + ", ".join(f"{k[0]}#{k[1]}={p:.2f}" for k, p in sorted(negs, key=lambda x: -x[1])))
    if name.startswith("2"):
        coef = m.named_steps["logisticregression"].coef_[0]
        order = np.argsort(-np.abs(coef))
        print("  most important features (+ = more smoke-like): "
              + ", ".join(f"{feats[i]} {coef[i]:+.2f}" for i in order[:6]))
    print()
