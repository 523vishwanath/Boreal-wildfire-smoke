import argparse
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/boreal_seg_960/data.yaml")
ap.add_argument("--model", default="yolo26m-seg.pt")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--batch", type=int, default=12)
ap.add_argument("--workers", type=int, default=14)
ap.add_argument("--project", default="/workspace/runs/boreal_seg")
ap.add_argument("--name", default="yolo26m_seg_960")
args = ap.parse_args()

model = YOLO(args.model)
model.train(
    data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
    workers=args.workers, cache="ram", device="0", project=args.project,
    name=args.name, exist_ok=True, patience=40, cos_lr=True, close_mosaic=15,
    seed=0, amp=True, plots=True,
    hsv_h=0.01, hsv_s=0.5, hsv_v=0.35, degrees=0.0, translate=0.1, scale=0.5,
    fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.0,
)

best = YOLO(model.trainer.best)
root = Path(args.data).parent
evals = [
    ("VAL", "data.yaml", "val", "val"),
    ("TEST: held-out videos (mostly SAM labels)", "data.yaml", "test", "test"),
    ("TEST: hand-drawn masks, held-out videos", "data_manual.yaml", "test", "test_manual"),
    ("Boreal official test (hand-drawn, but videos overlap train: reference only)", "data_boreal.yaml", "test", "boreal"),
]
for label, yaml, split, tag in evals:
    print(f"\n=== {label} ===")
    best.val(data=str(root / yaml), split=split, imgsz=args.imgsz, batch=8,
             project=args.project, name=f"{args.name}_{tag}", exist_ok=True)
