"""
Constant action: fill masked cells with a fixed color.
"""

from typing import List
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import Grid, Mask


CONTRACT = {
    "set_color": {"inputs": ["grid", "mask", "color"], "outputs": "grid", "mask_only": True,
                  "preconditions": ["color in 0..9"]},
}


def set_color(grid: Grid, mask: Mask, color: int) -> Grid:
    """
    Set all masked cells to the given color.

    Args:
        grid: input grid
        mask: boolean mask
        color: target color (0-9)

    Returns:
        New grid with masked cells set to color
    """
    validate_colors(grid)
    h, w = dims(grid)

    if not (0 <= color <= 9):
        raise ValueError(f"Invalid color {color}; must be 0-9")

    result = deepcopy_grid(grid)

    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                result[r][c] = color

    return result
