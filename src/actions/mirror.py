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
    Mirror horizontally (left-right flip) within mask using per-row bbox.

    For each row, finds left[r] and right[r] over masked cells.
    For each masked (r,c), computes c' = left[r] + right[r] - c.
    Writes grid[r][c] to result[r][c'] if (r,c') is also masked.
    Vacated masked cells become 0.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)

    # Process each row independently
    for r in range(h):
        # Find per-row mask bbox (left and right extent of masked cells in this row)
        left_r = None
        right_r = None
        for c in range(w):
            if mask[r][c]:
                if left_r is None:
                    left_r = c
                right_r = c

        # Skip rows with no masked cells
        if left_r is None:
            continue

        # First, set all masked cells in this row to 0 (default/vacated state)
        for c in range(w):
            if mask[r][c]:
                result[r][c] = 0

        # Then, for each masked cell, write its value to the mirror position if possible
        for c in range(w):
            if mask[r][c]:
                c_prime = left_r + right_r - c
                if 0 <= c_prime < w and mask[r][c_prime]:
                    # Write grid[r][c]'s value to result[r][c_prime]
                    result[r][c_prime] = grid[r][c]
                # else: drop (result[r][c] stays 0 from earlier initialization)

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

    # For each column, flip rows within bbox (only swap each pair once)
    mid_r = (r_min + r_max) // 2
    for c in range(c_min, c_max + 1):
        for r in range(r_min, mid_r + 1):
            mirror_r = r_min + (r_max - r)
            if r != mirror_r:  # Don't swap with self
                if mask[r][c] and mask[mirror_r][c]:
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

    # Transpose within bbox (only swap upper triangle to avoid double-swapping)
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            mirror_r = r_min + (c - c_min)
            mirror_c = c_min + (r - r_min)

            # Only swap if we're in upper triangle (r < mirror_r or r == mirror_r and c < mirror_c)
            if (r - r_min) < (c - c_min):  # Upper triangle
                if mask[r][c] and mask[mirror_r][mirror_c]:
                    result[r][c], result[mirror_r][mirror_c] = result[mirror_r][mirror_c], result[r][c]

    return result
