import csv, json, subprocess
from pathlib import Path
from collections import Counter, defaultdict

B = Path("/workspace/Boreal-Forest-Fire/Boreal-Forest-Fire-Subset-B")
vids = sorted(p for p in B.rglob("*") if p.suffix.lower() in {".mp4", ".mov"})
print(f"{len(vids)} videos, {sum(p.stat().st_size for p in vids) / 1e9:.1f} GB total")

by_folder = defaultdict(list)
for p in vids:
    by_folder[p.parent.name].append(p)
for folder, clips in sorted(by_folder.items()):
    print(f"  {folder}: {len(clips)} clips, e.g. {clips[0].name}")

def probe(p):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,codec_name:format=duration",
           "-of", "json", str(p)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)

print("\nSample clip per folder:")
for folder, clips in sorted(by_folder.items()):
    j = probe(clips[0])
    s = j["streams"][0]
    print(f"  {clips[0].name}: {s['width']}x{s['height']} {s['codec_name']} "
          f"fps={s['r_frame_rate']} duration={float(j['format']['duration']):.1f}s")

print("\nLabel files:")
for f in B.rglob("*.csv"):
    text = f.read_text(encoding="utf-8-sig")
    dialect = csv.Sniffer().sniff(text[:2000], delimiters=",;\t")
    rows = list(csv.reader(text.splitlines(), dialect))
    print(f"  {f.relative_to(B)}: {len(rows) - 1} rows")
    print(f"    header: {rows[0]}")
    for r in rows[1:4]:
        print(f"    {r}")
    for i, col in enumerate(rows[0]):
        vals = Counter(r[i] for r in rows[1:] if len(r) > i)
        if len(vals) <= 5:
            print(f"    column '{col}': {dict(vals)}")
