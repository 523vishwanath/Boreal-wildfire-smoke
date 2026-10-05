from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from eval_pixel import DATA, WEIGHTS, find, load, predict, scores

OUT = Path("/workspace/runs/boreal_seg/worst_cases")
OUT.mkdir(parents=True, exist_ok=True)
model = YOLO(WEIGHTS)

rows = []
for img in sorted((DATA / "images" / "test").glob("*.jpg")):
    pred = predict(model, img, 0.25)
    p = find("manual_masks", img.stem) or find("sam_masks", img.stem)
    if p:
        gt = load(p, pred.shape)
        rows.append((scores(pred, gt)[2], img, pred, gt))

rows.sort(key=lambda r: r[0])
for k, (iou, img, pred, gt) in enumerate(rows[:12]):
    im = cv2.imread(str(img))
    over = im.copy()
    over[pred & gt] = (0, 255, 255)    # yellow: correct
    over[gt & ~pred] = (0, 255, 0)     # green: missed smoke
    over[pred & ~gt] = (0, 0, 255)     # red: extra prediction
    out = cv2.addWeighted(im, 0.55, over, 0.45, 0)
    cv2.imwrite(str(OUT / f"{k:02d}_iou{iou:.2f}_{img.stem}.jpg"), np.hstack([im, out]))

print(f"Images with IoU < 0.3: {sum(r[0] < 0.3 for r in rows)}/{len(rows)}")
print(f"Saved worst 12 to {OUT}")
