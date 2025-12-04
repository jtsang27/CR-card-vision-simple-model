# ---------------------------------------------------------------------------
# Hand layout configuration (fractions of width/height)
# Tuned for 540x960 screenshots like hand_test.png
# ---------------------------------------------------------------------------

# Big band where the 4 cards live (relative to image height)
HAND_Y_FRAC_TOP = 0.82    # a bit above the card art
HAND_Y_FRAC_BOTTOM = 0.96 # a bit below the elixir cost

# Horizontal region spanning the 4 cards (relative to width)
HAND_X_FRAC_LEFT = 0.10   # left edge somewhere before card 0
HAND_X_FRAC_RIGHT = 0.90  # right edge somewhere after card 3

# How much to shrink inside each slot (to avoid overlap)
SLOT_MARGIN_FRAC_X = 0.08   # 8% margin left/right inside each slot
SLOT_MARGIN_FRAC_Y = 0.05   # 5% margin top/bottom inside the hand band


from typing import List, Tuple
import numpy as np
import cv2


def compute_hand_boxes(
    frame_shape: Tuple[int, int, int], num_slots: int = 4
) -> List[Tuple[int, int, int, int]]:
    """
    Compute (x1, y1, x2, y2) boxes for each hand slot, using the fractional
    constants above and horizontal+vertical margins so each crop tightly
    hugs its card.
    """
    h, w = frame_shape[:2]

    # full vertical band for the hand
    band_y1 = int(HAND_Y_FRAC_TOP * h)
    band_y2 = int(HAND_Y_FRAC_BOTTOM * h)

    # apply vertical margin inside band
    v_margin = int(SLOT_MARGIN_FRAC_Y * (band_y2 - band_y1))
    y1 = band_y1 + v_margin
    y2 = band_y2 - v_margin

    # full horizontal region for all 4 cards
    x_left = int(HAND_X_FRAC_LEFT * w)
    x_right = int(HAND_X_FRAC_RIGHT * w)

    total_width = x_right - x_left
    slot_width = total_width / float(num_slots)

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(num_slots):
        seg_x1 = x_left + i * slot_width
        seg_x2 = x_left + (i + 1) * slot_width

        # shrink horizontally inside this slot
        h_margin = SLOT_MARGIN_FRAC_X * slot_width
        x1 = int(seg_x1 + h_margin)
        x2 = int(seg_x2 - h_margin)

        boxes.append((x1, y1, x2, y2))

    return boxes