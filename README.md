# Wildfire smoke detection, segmentation and temporal alerts from drone video

Leak-free smoke detection and segmentation on the [Boreal Forest Fire](https://doi.org/10.1038/s41597-025-05634-0) UAV dataset, extended toward a video system that tracks smoke over time and raises alerts, with a web dashboard where you can upload a video.

**Headline results (held-out videos):**

| Task | Model | Result |
| --- | --- | --- |
| Detection | YOLO26m, 960 px | **0.933 mAP50 / 0.644 mAP50-95** |
| Segmentation | YOLO26m-seg, 960 px | **0.839 overall pixel IoU**; on par with SAM against hand-drawn masks (mean IoU 0.616 vs 0.628) |

Most of the work went into making these numbers trustworthy: a video-level split, catching leakage the official split has, auditing pseudo-label quality, and catching a camera-motion shortcut before training a temporal model.

## Dataset

Frames and 30-second clips from DJI Phantom 4 drone videos (4096 x 2160) of four prescribed burns in Finland: Evo, Heinola, Karkkila and Ruokolahti. One class: smoke.

| Subset | Content | Used for |
| --- | --- | --- |
| A | 4,954 frames with bounding boxes (256 smoke-free) | Detection |
| B | 288 video clips with one smoke / no-smoke label per clip (only 15 smoke-free) | Tracking, temporal analysis |
| C | 1,472 frames with SAM masks; 40 test frames with hand-drawn masks | Segmentation |

The dataset is not included here. Download it from [Fairdata](https://doi.org/10.23729/fd-72c6cf74-b8eb-3687-860d-bf93a1ab94c9) and see the dataset page for license terms.

## 1. Leak-free split

Neighbouring frames of a drone video are almost identical, so a random frame split leaks the test set into training. `detection/make_split.py`:

1. Groups frames by source video (from filenames such as `ruokolahti_DJI_0080_frame6`) and assigns whole videos to train / val / test per site, so every split covers all four fires.
2. Runs a perceptual-hash (dHash) near-duplicate check between splits.
3. Supports `--merge` for video files that are really one flight, and `--holdout-site` for leave-one-site-out tests.

The duplicate check flagged **140 of 1,014** test images: test video `karkkila_DJI_0008` matched train video `karkkila_DJI_0007` on 131 of 171 frames (one flight split into two files). After merging them, test near-duplicates dropped to **11 of 1,021**.

Final split: **3,038 / 895 / 1,021** images (train / val / test), no video in more than one split.

## 2. Detection

YOLO26m at 960 px, batch 16, cosine LR, 150 epochs with early stopping, small hue augmentation (smoke is grey), no vertical flips.

| Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| Val | 0.972 | 0.985 | 0.987 | 0.715 |
| **Test (held-out videos)** | **0.952** | **0.919** | **0.933** | **0.644** |

Inference is about 6.4 ms per image at 960 px on an RTX 4090.

**Fixing a 7% GPU utilisation bottleneck:** AutoBatch picked batch 4, and the CPU couldn't decode 4K JPEGs fast enough. Pre-resizing every image once to 960 px (`detection/resize_dataset.py`) and setting batch 16 cut epoch time from about 9 minutes to about 2.

## 3. Segmentation

**The official Subset C split leaks:** 11 of its 15 test videos also appear in train, and 7 validation videos overlap train (`segmentation/inspect_c.py`). Since every Subset C frame is also a Subset A frame, I reused the detection video split instead (`segmentation/prep_seg.py`), converted masks to YOLO polygons, used hand-drawn masks where available, and added 214 smoke-free frames as negatives.

**Label quality check:** SAM masks agree with hand-drawn masks at **mean IoU 0.646** (median 0.710, worst 0.278), so they are usable but noisy.

| Test set | Images | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Held-out videos (mostly SAM labels) | 288 | 0.868 | 0.542 | 0.615 | 0.469 |
| Hand-drawn masks, held-out videos | 11 | 0.661 | 0.727 | 0.570 | 0.304 |

Instance mAP understates quality for smoke, because SAM often splits one plume into several blobs. Pixel-level evaluation (`segmentation/eval_pixel.py`):

| Comparison | Smoke images | Mean IoU | Median IoU | Overall IoU |
| --- | --- | --- | --- | --- |
| Model vs labels, held-out videos | 270 | 0.646 | 0.883 | **0.839** |
| Model vs hand-drawn masks | 11 | 0.616 | 0.697 | 0.652 |
| SAM vs the same hand-drawn masks | 11 | 0.628 | 0.715 | |

The model matches its teacher (SAM) against human ground truth, while needing no prompts and running in about 13 ms per image. Smoke was predicted on 1 of 18 smoke-free test frames.

## 4. Temporal analysis (in progress)

**Goal:** tell smoke from look-alikes (fog, clouds) by how a region behaves over time: smoke grows from a fixed source and churns; fog tends to drift as one block.

**Done so far:**

- **Tracking:** YOLO26m-seg + BoT-SORT on Subset B clips, shrunk to 960 px at 4 fps (12.4 GB to 228 MB for Evo). Tuned for weak detections; note that Ultralytics scales `track_buffer` by fps / 30. 91 tracks on 31 Evo clips, the main plume usually tracked for the whole clip.
- **Automatic track labels:** matched clip frames to Subset A's human-annotated frames (786 of 931 matched) and voted per track: 53 smoke, 10 false alarm. The matching also maps every clip to its source drone video, so clips inherit the leak-free split.
- **Behaviour features** with drone motion removed: growth, base drift, optical-flow coherence and entropy inside the mask, brightness relative to surroundings, and motion relative to a background ring.

**Findings:**

| Finding | Evidence |
| --- | --- |
| Persistence alone doesn't stop false alarms | A fog clip (FP34) and 4 of 15 smoke-free Boreal clips trigger both a frame rule and a "track lasts 4 s" rule |
| The detector confuses fog and clouds with smoke out of domain | 13 of 20 fog / low-cloud photos flagged at conf 0.25, 7 of 20 at 0.5 |
| **Motion features learned drone speed (a shortcut)** | Smoke filmed at the same camera speed as the fog clip drifts just as much (base drift 0.044 vs 0.042); camera speed vs drift correlation 0.59 |
| Consistent separators across all false-alarm types | Smoke has higher confidence, churns more (lower flow coherence, higher entropy) and is brighter than its surroundings |
| With 12 false-alarm tracks, track confidence beats behaviour features | Leave-one-video-out AUC: confidence 0.758, behaviour 0.649, behaviour + drift 0.671 |

The next step is more negative video (fog and cloud footage, more Boreal sites) before training a GRU / temporal CNN.

## 5. Dashboard

`dashboard/app.py` is a Gradio app: upload a video and get a verdict, an annotated video (masks, track IDs, live scores), an alert log with snapshots, a smoke-score timeline and a table of tracked regions.

Current alert rule: a track alerts when it has existed for at least 3 s, was visible in at least 60% of those frames, and its mean confidence over the last 2 s is at least 0.6 (adjustable in the UI). The temporal model will replace this rule once it beats it on held-out data.

```bash
pip install -r requirements.txt
# download the weights into weights/ (see below), then:
cd dashboard && python app.py
```

## Weights

Weights are not stored in git. Download them from this repo's **Releases** page into `weights/`:

- `yolo26m_seg_smoke.pt`: segmentation model used by the dashboard
- `yolo26m_det_smoke.pt`: detection model

Or point the dashboard at your own file: `SMOKE_WEIGHTS=/path/to/best.pt python app.py`.

## Repository layout

```
detection/      make_split.py, resize_dataset.py, train_det.py
segmentation/   inspect_c.py, prep_seg.py, train_seg.py, resume_seg.py, eval_pixel.py, worst_cases.py
temporal/       inventory_b.py, track_clips.py, run_video.py, auto_label.py,
                extract_features.py, compare_features.py, train_baseline.py, botsort_smoke.yaml
dashboard/      smoke_pipeline.py, app.py
results/        metrics, curves, split manifest, track labels, feature tables
weights/        trained models (downloaded separately)
```

The scripts were run on a RunPod RTX 4090 pod and use `/workspace/...` paths; adjust them to your own data locations.

## Limitations

- Temporal results so far use one site (Evo) and few false-alarm examples; numbers are early signals.
- Evaluation sets have few smoke-free frames, so false-alarm rates are not well measured yet.
- Subset B labels are per clip and were not validated by the dataset authors.

## Citation

Pesonen, J., Raita-Hakola, A.-M., Joutsalainen, J., et al. (2025). *Boreal Forest Fire: UAV-collected Wildfire Detection and Smoke Segmentation Dataset.* Scientific Data, 12, 1419. https://doi.org/10.1038/s41597-025-05634-0

## Author

Vishwanath Reddy Ninganolla · [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)
