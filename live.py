"""Minimal OpenCV live inference script.

Use this when you want a quick camera demo without launching the full Tkinter
console in app.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight webcam inference.")
    parser.add_argument("--model", type=Path, default=Path("best.pt"), help="Path to YOLO model weights.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    return parser.parse_args()


def draw_counts(frame, counts: Counter[str], confidence_threshold: float) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (8, 12, 16), -1)
    cv2.putText(
        frame,
        f"Weapon Detection | threshold {confidence_threshold:.0%} | press q to quit",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (244, 247, 251),
        2,
        cv2.LINE_AA,
    )

    y_offset = 78
    if not counts:
        cv2.putText(frame, "No detections above threshold", (14, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 190, 200), 2)
        return

    for label, count in counts.most_common():
        cv2.putText(
            frame,
            f"{label}: {count}",
            (14, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (85, 194, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 32


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=args.conf, verbose=False)[0]
            annotated_frame = result.plot()

            labels = []
            if result.boxes is not None:
                for cls_id, confidence in zip(result.boxes.cls, result.boxes.conf):
                    if float(confidence) >= args.conf:
                        labels.append(model.names[int(cls_id)])

            draw_counts(annotated_frame, Counter(labels), args.conf)
            cv2.imshow("Weapon Detection - Webcam", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
