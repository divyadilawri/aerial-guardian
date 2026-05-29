import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict, deque

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import supervision as sv


# ── Config ────────────────────────────────────────────────────────────────────
PERSON_CLASS_ID  = 0
CONF_THRESHOLD   = 0.25
IOU_THRESHOLD    = 0.45
SLICE_HW         = 640
OVERLAP_RATIO    = 0.2
TAIL_LENGTH      = 30
# ─────────────────────────────────────────────────────────────────────────────


def build_sahi_model(weights: str, device: str):
    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights,
        confidence_threshold=CONF_THRESHOLD,
        device=device,
    )


def sahi_detect(sahi_model, frame: np.ndarray):
    result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=SLICE_HW,
        slice_width=SLICE_HW,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMM",
        postprocess_match_threshold=IOU_THRESHOLD,
        verbose=0,
    )
    boxes, confs, classes = [], [], []
    for pred in result.object_prediction_list:
        if pred.category.id != PERSON_CLASS_ID:
            continue
        b = pred.bbox
        boxes.append([b.minx, b.miny, b.maxx, b.maxy])
        confs.append(pred.score.value)
        classes.append(PERSON_CLASS_ID)

    if not boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    return np.array(boxes), np.array(confs), np.array(classes, dtype=int)


def run(video_path: str, weights: str, output_path: str, device: str):
    cap = cv2.VideoCapture(video_path)
    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 25
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_in, (W, H)
    )

    sahi_model = build_sahi_model(weights, device)

    tracker = sv.ByteTrack(
        track_activation_threshold=CONF_THRESHOLD,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8,
        frame_rate=int(fps_in),
    )

    box_ann   = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    tails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TAIL_LENGTH))

    frame_times = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()

        xyxy, confs, classes = sahi_detect(sahi_model, frame)

        dets = sv.Detections(
            xyxy=xyxy,
            confidence=confs,
            class_id=classes,
        ) if len(xyxy) else sv.Detections.empty()

        dets = tracker.update_with_detections(dets)

        for xyxy_box, tid in zip(dets.xyxy, dets.tracker_id if dets.tracker_id is not None else []):
            cx = int((xyxy_box[0] + xyxy_box[2]) / 2)
            cy = int((xyxy_box[1] + xyxy_box[3]) / 2)
            tails[tid].append((cx, cy))

        annotated = frame.copy()
        for tid, pts in tails.items():
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                alpha = i / len(pts_list)
                color = (0, int(255 * alpha), int(200 * alpha))
                cv2.line(annotated, pts_list[i-1], pts_list[i], color, 2)

        labels = [f"ID {tid}" for tid in (dets.tracker_id if dets.tracker_id is not None else [])]
        annotated = box_ann.annotate(annotated, dets)
        annotated = label_ann.annotate(annotated, dets, labels=labels)

        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
        cur_fps = 1.0 / (sum(frame_times[-30:]) / min(len(frame_times), 30))
        cv2.putText(annotated, f"FPS: {cur_fps:.1f}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}  |  tracks: {len(dets)}  |  FPS: {cur_fps:.1f}")

    cap.release()
    writer.release()
    avg_fps = len(frame_times) / sum(frame_times)
    print(f"\nDone. {frame_idx} frames | avg FPS: {avg_fps:.2f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   required=True,  help="Input video path")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights")
    parser.add_argument("--output",  default="output.mp4")
    parser.add_argument("--device",  default="cpu", help="cuda / cpu / mps")
    args = parser.parse_args()
    run(args.video, args.weights, args.output, args.device)