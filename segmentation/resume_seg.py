from pathlib import Path
from ultralytics import YOLO

RUN = Path("/workspace/runs/boreal_seg/yolo26m_seg_960")
DATA = Path("/root/boreal_seg_960")

model = YOLO(RUN / "weights" / "last.pt")
model.train(resume=True)   # picks up the same settings and continues from the last saved epoch

best = YOLO(model.trainer.best)
evals = [
    ("VAL", "data.yaml", "val", "val"),
    ("TEST: held-out videos (mostly SAM labels)", "data.yaml", "test", "test"),
    ("TEST: hand-drawn masks, held-out videos", "data_manual.yaml", "test", "test_manual"),
    ("Boreal official test (sanity check only, not a real result)", "data_boreal.yaml", "test", "boreal"),
]
for label, yaml, split, tag in evals:
    print(f"\n=== {label} ===")
    best.val(data=str(DATA / yaml), split=split, imgsz=960, batch=8,
             project=str(RUN.parent), name=f"{RUN.name}_{tag}", exist_ok=True)
