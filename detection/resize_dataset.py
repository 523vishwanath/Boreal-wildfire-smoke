import shutil, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

SRC = Path("/workspace/boreal_yolo")
DST = Path("/workspace/boreal_yolo_960")
LONG = 960

def resize_one(src):
    dst = (DST / "images" / src.relative_to(SRC / "images")).with_suffix(".jpg")
    with Image.open(src) as im:
        im.draft("RGB", (LONG, LONG))          # fast JPEG decode at reduced size
        im = im.convert("RGB")
        im.thumbnail((LONG, LONG), Image.Resampling.LANCZOS)
        im.save(dst, quality=95)
    return 1

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if DST.exists():
        shutil.rmtree(DST)
    for s in ("train", "val", "test"):
        (DST / "images" / s).mkdir(parents=True)
        shutil.copytree(SRC / "labels" / s, DST / "labels" / s)   # copies real files, not symlinks
    files = [p for p in (SRC / "images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"Resizing {len(files)} images with {workers} workers...")
    with ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(resize_one, files, chunksize=16))
    yaml = (SRC / "data.yaml").read_text().replace(str(SRC.resolve()), str(DST.resolve()))
    (DST / "data.yaml").write_text(yaml)
    print(f"Done: {n} images -> {DST}")
