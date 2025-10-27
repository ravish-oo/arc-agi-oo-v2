"""
Mirror actions: reflect grid within mask.

All mirrors operate only on cells inside the mask.
"""

from typing import List
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import get_mask_bbox, Grid, Mask


CONTRACT = {
    "mirror_h": {"inputs": ["grid", "mask"], "outputs": "grid", "mask_only": True,
                 "preconditions": ["colors in 0..9"], "policies": ["zero-pad", "deterministic"]},
    "mirror_v": {"inputs": ["grid", "mask"], "outputs": "grid", "mask_only": True,
                 "preconditions": ["colors in 0..9"], "policies": ["zero-pad", "deterministic"]},
    "mirror_diag": {"inputs": ["grid", "mask"], "outputs": "grid", "mask_only": True,
                    "preconditions": ["colors in 0..9", "mask_bbox_square"], "policies": ["zero-pad", "deterministic"]},
}


def mirror_h(grid: Grid, mask: Mask) -> Grid:
    """
    Mirror horizontally (left-right flip) within mask.

    For each masked row, reflects columns within the mask's bbox.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)
    r_min, r_max, c_min, c_max = get_mask_bbox(mask)

    if r_max < r_min:  # Empty mask
        return result

    # For each row, flip columns within bbox
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if mask[r][c]:
                # Mirror column: c -> c_min + (c_max - c)
                mirror_c = c_min + (c_max - c)
                if mask[r][mirror_c]:
                    # Swap values
                    result[r][c], result[r][mirror_c] = result[r][mirror_c], result[r][c]

    return result


def mirror_v(grid: Grid, mask: Mask) -> Grid:
    """
    Mirror vertically (top-bottom flip) within mask.

    For each masked column, reflects rows within the mask's bbox.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)
    r_min, r_max, c_min, c_max = get_mask_bbox(mask)

    if r_max < r_min:  # Empty mask
        return result

    # For each column, flip rows within bbox
    for c in range(c_min, c_max + 1):
        for r in range(r_min, r_max + 1):
            if mask[r][c]:
                # Mirror row: r -> r_min + (r_max - r)
                mirror_r = r_min + (r_max - r)
                if mask[mirror_r][c]:
                    # Swap values
                    result[r][c], result[mirror_r][c] = result[mirror_r][c], result[r][c]

    return result


def mirror_diag(grid: Grid, mask: Mask) -> Grid:
    """
    Mirror along main diagonal (y=x) within mask.

    Requires: mask bbox must be square.
    Raises ValueError if bbox is not square.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)
    r_min, r_max, c_min, c_max = get_mask_bbox(mask)

    if r_max < r_min:  # Empty mask
        return result

    # Check bbox is square
    bbox_h = r_max - r_min + 1
    bbox_w = c_max - c_min + 1
    if bbox_h != bbox_w:
        raise ValueError(f"mirror_diag requires square mask bbox; got {bbox_h}×{bbox_w}")

    # Transpose within bbox
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if mask[r][c]:
                # Map (r, c) -> (r_min + (c - c_min), c_min + (r - r_min))
                mirror_r = r_min + (c - c_min)
                mirror_c = c_min + (r - r_min)
                if mask[mirror_r][mirror_c]:
                    # Swap values
                    result[r][c], result[mirror_r][mirror_c] = result[mirror_r][mirror_c], result[r][c]

    return result
