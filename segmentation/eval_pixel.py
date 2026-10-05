import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

C = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-C")
DATA = Path("/root/boreal_seg_960")
WEIGHTS = "/workspace/runs/boreal_seg/yolo26m_seg_960/weights/best.pt"

def find(kind, stem):
    for sub in ("train", "valid", "test"):
        p = C / kind / sub / f"{stem}.png"
        if p.exists():
            return p
    return None

def load(p, shape):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 127

def predict(model, img, conf):
    r = model.predict(str(img), imgsz=960, conf=conf, retina_masks=True, verbose=False)[0]
    h, w = r.orig_shape
    if r.masks is None:
        return np.zeros((h, w), bool)
    return (r.masks.data.cpu().numpy() > 0.5).any(0)

def scores(pred, gt):
    i, u = (pred & gt).sum(), (pred | gt).sum()
    return i, u, i / max(u, 1), 2 * i / max(pred.sum() + gt.sum(), 1)

def evaluate(model, folder, manual_only, conf):
    I = U = 0
    ious, dices, false_alarms, sam_ious = [], [], [], []
    for img in sorted((DATA / "images" / folder).glob("*.jpg")):
        pred = predict(model, img, conf)
        gt_p = find("manual_masks", img.stem) if manual_only else (find("manual_masks", img.stem) or find("sam_masks", img.stem))
        gt = load(gt_p, pred.shape) if gt_p else np.zeros(pred.shape, bool)
        if not gt.any():
            false_alarms.append(pred.any())
            continue
        i, u, iou, dice = scores(pred, gt)
        I, U = I + i, U + u
        ious.append(iou)
        dices.append(dice)
        if manual_only and (s := find("sam_masks", img.stem)):
            sam_ious.append(scores(load(s, gt.shape), gt)[2])

    print(f"  smoke images: {len(ious)}")
    print(f"  model IoU   per-image mean {np.mean(ious):.3f} | median {np.median(ious):.3f} | overall {I / max(U, 1):.3f}")
    print(f"  model Dice  per-image mean {np.mean(dices):.3f}")
    if sam_ious:
        print(f"  SAM IoU vs same hand-drawn masks: mean {np.mean(sam_ious):.3f} | median {np.median(sam_ious):.3f}")
    if false_alarms:
        print(f"  false alarms: smoke predicted on {sum(false_alarms)}/{len(false_alarms)} smoke-free frames")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    model = YOLO(WEIGHTS)
    print(f"conf threshold: {args.conf}")
    print("\n=== Held-out test videos (labels: hand-drawn where available, else SAM) ===")
    evaluate(model, "test", manual_only=False, conf=args.conf)
    print("\n=== Hand-drawn test frames: model vs SAM, both scored against the human masks ===")
    evaluate(model, "test_manual", manual_only=True, conf=args.conf)
