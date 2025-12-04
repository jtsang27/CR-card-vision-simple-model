"""
card_detection.py

Card detection for Clash Royale using a locally trained YOLOv8 model
on the *split hand* card crops.

Pipeline:
    full frame  -->  crop_hand_slots()  -->  YOLO on each crop

Usage (CLI test):
    python card_detection.py hand_test.png

Usage (imported):
    from card_detection import CardDetector
    detector = CardDetector()
    cards = detector.detect(frame)  # frame is full BGR screenshot

Returns:
    [
        {"slot": 0, "name": "...", "confidence": ...},
        {"slot": 1, "name": "...", "confidence": ...},
        {"slot": 2, "name": "...", "confidence": ...},
        {"slot": 3, "name": "...", "confidence": ...},
    ]
"""

from typing import List, Dict, Tuple

import cv2
import numpy as np

from split_hand_cards import crop_hand_slots
from local_yolo_model import load_local_model


class CardDetector:
    def __init__(self) -> None:
        # Load local YOLO model once
        self.model = load_local_model()
        # threshold for deciding "unknown" vs a real label
        self.min_confidence = 0.20

    def _predict_single_card(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Run YOLO on a single cropped card image and return (class_name, confidence).
        Assumes your YOLO model was trained on similar card-tile crops.
        """
        # YOLO will internally resize img to its training size (e.g., 640x640)
        results = self.model(img, conf=0.01, verbose=False)[0]

        if len(results.boxes) == 0:
            return "unknown", 0.0

        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(confs.argmax())
        conf = float(confs[best_idx])
        cls_id = int(clss[best_idx])

        name = self.model.names[cls_id]

        if conf < self.min_confidence:
            return "unknown", conf

        return name, conf

    def detect(self, frame: np.ndarray) -> List[Dict[str, float]]:
        """
        Detect the cards currently in hand from a *full* frame.

        Args:
            frame: BGR numpy array from cv2 (full Clash Royale screenshot).

        Returns:
            List of dicts with "slot" (0..3), "name", and "confidence".
        """
        crops, _ = crop_hand_slots(frame)

        results: List[Dict[str, float]] = []
        for i, crop in enumerate(crops):
            name, conf = self._predict_single_card(crop)
            results.append(
                {
                    "slot": i,
                    "name": name,
                    "confidence": conf,
                }
            )

        return results


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Clash Royale hand card detection on a screenshot (local YOLO on split hand)."
    )
    parser.add_argument(
        "image_path",
        help="Path to a full Clash Royale screenshot.",
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