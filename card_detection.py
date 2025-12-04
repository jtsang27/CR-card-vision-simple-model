"""
card_detection.py

Card detection for Clash Royale using a locally trained YOLOv8 model
on the FULL frame, then mapping detections into 4 hand slots.

Usage (CLI test):
    python card_detection.py hand_test.png

Usage (imported):
    from card_detection import CardDetector
    detector = CardDetector()
    cards = detector.detect(frame)  # frame is full BGR screenshot

Returns:
    [
        {"slot": 0, "name": "Hog Rider", "confidence": 0.91},
        {"slot": 1, "name": "Ice Spirit", "confidence": 0.88},
        {"slot": 2, "name": "Log", "confidence": 0.93},
        {"slot": 3, "name": "Fireball", "confidence": 0.90},
    ]
"""

from typing import List, Dict, Tuple

import cv2
import numpy as np

from split_hand_cards import compute_hand_boxes
from local_yolo_model import load_local_model


class CardDetector:
    def __init__(self) -> None:
        # Load local YOLO model once
        self.model = load_local_model()
        # Threshold for treating a detection as "real"
        self.min_confidence = 0.3

    def detect(self, frame: np.ndarray) -> List[Dict[str, str]]:
        """
        Detect cards currently in hand from a *full* frame.

        Args:
            frame: BGR numpy array from cv2 (full Clash Royale screenshot/frame).

        Returns:
            List of dicts with "slot" (0..3), "name", and "confidence".
        """
        # Run YOLO once on the full frame
        results = self.model(frame, conf=0.1, verbose=False)[0]

        # Precompute the 4 hand-slot rectangles
        slot_boxes = compute_hand_boxes(frame.shape, num_slots=4)
        num_slots = len(slot_boxes)

        # Initialize per-slot best detection
        slots: List[Dict[str, float | str]] = [
            {"slot": i, "name": "unknown", "confidence": 0.0}
            for i in range(num_slots)
        ]

        if len(results.boxes) == 0:
            return slots

        xyxy = results.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = results.boxes.conf.cpu().numpy()  # (N,)
        clss = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            # center of the detection box
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)

            # assign to a slot if center falls inside that slot's rect
            for i, (sx1, sy1, sx2, sy2) in enumerate(slot_boxes):
                if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                    if conf > slots[i]["confidence"]:
                        name = self.model.names[cls_id]
                        slots[i]["name"] = name
                        slots[i]["confidence"] = float(conf)
                    break  # center won't be in multiple slots

        # Apply global min_confidence threshold
        for s in slots:
            if s["confidence"] < self.min_confidence:
                s["name"] = "unknown"

        return slots


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Clash Royale hand card detection on a screenshot (local YOLO, full-frame)."
    )
    parser.add_argument(
        "image_path",
        help="Path to a full Clash Royale screenshot (same resolution as your capture loop).",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image_path}")

    detector = CardDetector()
    detections = detector.detect(img)
    print("Detections:")
    for d in detections:
        print(
            f"  Slot {d['slot']}: {d['name']} (conf={d['confidence']:.3f})"
        )


if __name__ == "__main__":
    _cli()