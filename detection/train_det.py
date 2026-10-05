#!/usr/bin/env python3
"""
Smoke detection training on the Boreal split (Ultralytics YOLO26).
Trains, then evaluates the best checkpoint on val and on the held-out test split.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from make_split.py")
    ap.add_argument("--model", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = auto batch (~60%% GPU memory)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/workspace/runs/boreal")
    ap.add_argument("--name", default="yolo26m_960")
    args = ap.parse_args()

    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    cache = False if args.cache == "none" else args.cache

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        workers=args.workers,
        cache=cache,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=30,
        cos_lr=True,
        close_mosaic=15,
        seed=0,
        amp=True,
        plots=True,
        # augmentation: smoke is gray/white, so keep hue shifts small
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.35,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        mosaic=1.0,
        mixup=0.0,
    )

    best = model.trainer.best
    print(f"\nBest weights: {best}")

    best_model = YOLO(best)
    print("\n=== VAL ===")
    best_model.val(data=args.data, split="val", imgsz=args.imgsz, project=args.project,
                   name=f"{args.name}_val", exist_ok=True)
    print("\n=== TEST (held-out videos) ===")
    best_model.val(data=args.data, split="test", imgsz=args.imgsz, project=args.project,
                   name=f"{args.name}_test", exist_ok=True)


if __name__ == "__main__":
    main()
