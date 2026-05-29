# Aerial Guardian — Drone Person Detection & Tracking

## Setup Instructions

### Requirements
- macOS with Apple Silicon (tested on M4) or any machine with CUDA GPU
- Python 3.11
- ~2GB disk space

### Installation

```bash
git clone <your-repo-url>
cd aerial-guardian
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install ultralytics sahi supervision opencv-python numpy
```

### Prepare Dataset
Download VisDrone2019-MOT-val and place sequences inside `data/`.
Convert an image sequence to video:

```bash
python3 -c "
import cv2, glob
seq = 'data/VisDrone2019-MOT-val/sequences/uav0000339_00001_v'
frames = sorted(glob.glob(seq + '/*.jpg'))
img = cv2.imread(frames[0])
h, w = img.shape[:2]
out = cv2.VideoWriter('data/input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
for f in frames: out.write(cv2.imread(f))
out.release()
"
```

### Run the Pipeline

```bash
python detect_track.py \
  --video data/input.mp4 \
  --weights yolov8n.pt \
  --output data/output.mp4 \
  --device mps        # use 'cuda' for NVIDIA, 'cpu' as fallback
```

---

## Summary Report

### Architecture & Small Object Detection

The pipeline uses **YOLOv8-nano** as the base detector combined with **SAHI**
(Slicing Aided Hyper Inference). VisDrone persons can be as small as 5×10 pixels
in a full-resolution frame — a single 640px inference pass misses them entirely.
SAHI divides each frame into overlapping 640×640 tiles, runs YOLOv8n on each tile
independently, then merges results using Non-Maximum Merging (NMM) to eliminate
cross-tile duplicate detections. This recovers small targets with no retraining required.

Tile config: 640×640 tiles, 20% overlap, NMM threshold 0.45 IoU.

### Handling ID Switching from Drone Ego-Motion

Drone movement causes rapid background shifts and target occlusions, making ID
assignment challenging. We address this with **ByteTrack**, which offers three
key advantages:

1. It uses both high- and low-confidence detections. Partially occluded persons
   produce weak detections that other trackers discard — ByteTrack uses them to
   maintain track continuity.
2. A Kalman filter predicts each track's position across frames, compensating for
   ego-motion between detections.
3. `lost_track_buffer=30` keeps a track alive for 30 frames after the last
   detection, bridging short occlusions without reassigning IDs.
4. A high matching threshold (0.8 IoU) prevents premature ID reassignment when
   two targets are close together.

### FPS Benchmark

| Hardware         | Model     | Mode       | Avg FPS |
|-----------------|-----------|------------|---------|
| MacBook Pro M4   | YOLOv8n   | SAHI tiles | 8.15    |

Tested on: 275 frames, 1920×1080 resolution, sequence uav0000339_00001_v.
SAHI tiling generates ~6 sub-images per frame at this resolution, explaining
the lower FPS compared to single-pass inference (~25 FPS).

### Edge Deployment — NVIDIA Jetson Adaptation

To deploy on a Jetson Orin Nano or similar edge hardware:

1. **Export to TensorRT**: `yolo export model=yolov8n.pt format=engine device=0`
   — typically gives 2–3× speedup over PyTorch on Jetson.
2. **INT8 quantisation**: Calibrate on VisDrone val samples for <2% mAP drop
   with another 2× speed gain.
3. **Reduce tile count**: Use 480×480 tiles or reduce overlap to 10% to cut
   the number of sub-images per frame from ~6 to ~4.
4. **Frame decimation**: Run detection at 15 FPS input and interpolate bounding
   boxes between frames for smooth 30 FPS output video.
5. **Target**: Jetson Orin Nano can sustain ~10–15 FPS with TensorRT + INT8
   on 1080p drone footage with these optimisations.

### Engineering Trade-offs

- **SAHI vs single-pass**: SAHI recovers small targets but costs ~3–4× latency.
  For real-time edge use, single-pass with a larger input resolution (1280px)
  is a better trade-off.
- **YOLOv8n vs YOLOv8s**: nano runs at 8 FPS; small runs at ~5 FPS but detects
  more targets. For submission we chose nano to prioritise speed.
- **ByteTrack lost buffer**: A longer buffer (60 frames) reduces ID switches but
  risks keeping ghost tracks. 30 frames is the sweet spot for drone footage.