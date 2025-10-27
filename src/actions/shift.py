"""
Shift action: translate masked cells by (dr, dc).

Cells shifted out of bounds or out of mask are dropped.
Vacated masked cells are set to 0.
"""

from typing import List
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import Grid, Mask


CONTRACT = {
    "shift": {"inputs": ["grid", "mask", "dr", "dc"], "outputs": "grid", "mask_only": True,
              "preconditions": ["colors in 0..9"], "policies": ["zero-pad", "deterministic"]},
}


def shift(grid: Grid, mask: Mask, dr: int, dc: int) -> Grid:
    """
    Shift masked cells by (dr, dc).

    - Cells shifted to positions outside grid bounds or outside mask: dropped
    - Vacated positions that were masked: set to 0
    - Non-masked cells: unchanged
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)

    # First, clear vacated masked cells to 0
    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                result[r][c] = 0

    # Then, copy shifted values
    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                # Destination position
                nr = r + dr
                nc = c + dc

                # Check if destination is in bounds and in mask
                if 0 <= nr < h and 0 <= nc < w and mask[nr][nc]:
                    result[nr][nc] = grid[r][c]

    return result
