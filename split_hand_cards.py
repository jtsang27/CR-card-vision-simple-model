"""
split_hand_cards.py

Given a full Clash Royale screenshot (phone/emulator mirrored on your screen),
this script automatically crops out the 4 cards in your hand into separate
images like the example you showed (about 140x176, depending on resolution).

Usage (standalone):
    python split_hand_cards.py input_screenshot.png

Output:
    cards/
      input_screenshot_slot1.png
      input_screenshot_slot2.png
      input_screenshot_slot3.png
      input_screenshot_slot4.png

You can also import this from your vision code:

    from split_hand_cards import crop_hand_slots
    crops, boxes = crop_hand_slots(frame)

    # crops -> list of 4 small images (numpy arrays)
    # boxes -> list of (x1, y1, x2, y2) in the original frame
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Hand layout configuration (fractions of width/height)
# ---------------------------------------------------------------------------
# These numbers assume something like a 720x1280 capture with the phone in
# portrait and mirrored on a landscape monitor. They DEFINITELY might need
# tweaking for your setup – that's why the script also saves a *_boxes.png
# image so you can visually check alignment.

HAND_Y_FRAC_TOP = 0.83   # 0.81 * 960 ≈ 778
HAND_Y_FRAC_BOTTOM = 0.95  # 0.95 * 960 ≈ 912

# Horizontal region spanning the 4 main cards (excluding the "Next" card)
# Roughly columns ~90–450 out of 540
HAND_X_FRAC_LEFT = 0.23   # move left edge right
HAND_X_FRAC_RIGHT = 0.96


def compute_hand_boxes(
    frame_shape: Tuple[int, int, int], num_slots: int = 4
) -> List[Tuple[int, int, int, int]]:
    """
    Compute (x1, y1, x2, y2) boxes for each hand slot, using the fractional
    constants above and a horizontal margin so there is space between crops.
    """
    h, w = frame_shape[:2]

    y1 = int(HAND_Y_FRAC_TOP * h)
    y2 = int(HAND_Y_FRAC_BOTTOM * h)

    x_left = int(HAND_X_FRAC_LEFT * w)
    x_right = int(HAND_X_FRAC_RIGHT * w)

    total_width = x_right - x_left
    slot_width = total_width / float(num_slots)

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(num_slots):
        # segment for this slot before margin
        seg_x1 = x_left + i * slot_width
        seg_x2 = x_left + (i + 1) * slot_width

        # shrink inside that segment to leave horizontal space
        x1 = int(seg_x1)
        x2 = int(seg_x2)

        boxes.append((x1, y1, x2, y2))

    return boxes

def crop_hand_slots(
    frame: np.ndarray, num_slots: int = 4
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Given a full game frame, return:
      - crops: list of small images for each card slot
      - boxes: list of (x1, y1, x2, y2) rectangles in the original frame
    """
    boxes = compute_hand_boxes(frame.shape, num_slots=num_slots)

    crops: List[np.ndarray] = []
    for (x1, y1, x2, y2) in boxes:
        crop = frame[y1:y2, x1:x2].copy()
        crops.append(crop)

    return crops, boxes


def draw_boxes(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Draw green rectangles and slot indices on the frame for debugging.
    """
    img = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(i),
            (x1 + 5, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return img


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop Clash Royale hand cards from a full screenshot."
    )
    parser.add_argument(
        "image_path",
        help="Path to a full Clash Royale screenshot (same resolution as your capture).",
    )
    args = parser.parse_args()

    input_path = Path(args.image_path)
    if not input_path.exists():
        raise SystemExit(f"Input image does not exist: {input_path}")

    frame = cv2.imread(str(input_path))
    if frame is None:
        raise SystemExit(f"Could not read image: {input_path}")

    crops, boxes = crop_hand_slots(frame)

    # Create output directory next to the input
    out_dir = input_path.parent / "cards"
    out_dir.mkdir(exist_ok=True)

    stem = input_path.stem
    for i, crop in enumerate(crops, start=1):
        out_path = out_dir / f"{stem}_slot{i}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved card slot {i}: {out_path}")

    # Save debug image with boxes
    debug_img = draw_boxes(frame, boxes)
    debug_path = input_path.parent / f"{stem}_boxes.png"
    cv2.imwrite(str(debug_path), debug_img)
    print(f"Saved debug (with boxes): {debug_path}")

    print("\nIf the green boxes don't line up with your hand cards,")
    print("open the *_boxes.png image and tweak the HAND_*_FRAC")
    print("constants at the top of this script, then run again.")


if __name__ == "__main__":
    _cli()