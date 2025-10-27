"""
Constructor actions: draw geometric shapes.

draw_box_on_components: draws rectangular borders around mask components
draw_line_axis_aligned: draws lines along axis at bbox minimum
"""

from typing import List, Optional
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import find_e4_components, get_mask_bbox, Grid, Mask


CONTRACT = {
    "draw_box_on_components": {"inputs": ["grid", "mask", "thickness", "color?"], "outputs": "grid", "mask_only": True,
                                "preconditions": ["colors in 0..9", "thickness>=1"]},
    "draw_line_axis_aligned": {"inputs": ["grid", "mask", "axis", "color"], "outputs": "grid", "mask_only": True,
                                "preconditions": ["colors in 0..9", "axis in {row,col}"]},
}


def draw_box_on_components(grid: Grid, mask: Mask, thickness: int = 1,
                            color: Optional[int] = None) -> Grid:
    """
    Draw rectangular border around each E4-connected component of the mask.

    Args:
        grid: input grid
        mask: boolean mask
        thickness: border thickness (default 1)
        color: color to paint border (if None, keep existing colors)

    Returns:
        New grid with borders drawn on components
    """
    validate_colors(grid)
    h, w = dims(grid)

    if color is not None and not (0 <= color <= 9):
        raise ValueError(f"Invalid color {color}; must be 0-9")

    result = deepcopy_grid(grid)
    components = find_e4_components(mask)

    for component in components:
        if not component:
            continue

        # Get bbox of component
        r_coords = [r for r, c in component]
        c_coords = [c for r, c in component]
        r_min, r_max = min(r_coords), max(r_coords)
        c_min, c_max = min(c_coords), max(c_coords)

        # Draw border of given thickness
        for t in range(thickness):
            # Top and bottom edges
            for c in range(max(0, c_min - t), min(w, c_max + t + 1)):
                # Top edge
                if 0 <= r_min - t < h and mask[r_min - t][c]:
                    if color is not None:
                        result[r_min - t][c] = color
                # Bottom edge
                if 0 <= r_max + t < h and mask[r_max + t][c]:
                    if color is not None:
                        result[r_max + t][c] = color

            # Left and right edges
            for r in range(max(0, r_min - t), min(h, r_max + t + 1)):
                # Left edge
                if 0 <= c_min - t < w and mask[r][c_min - t]:
                    if color is not None:
                        result[r][c_min - t] = color
                # Right edge
                if 0 <= c_max + t < w and mask[r][c_max + t]:
                    if color is not None:
                        result[r][c_max + t] = color

    return result


def draw_line_axis_aligned(grid: Grid, mask: Mask, axis: str, color: int) -> Grid:
    """
    Draw axis-aligned line through mask bbox minimum.

    Args:
        grid: input grid
        mask: boolean mask
        axis: "row" or "col"
        color: line color (0-9)

    Returns:
        New grid with line drawn

    For axis="row": draws horizontal line at r=r_min of bbox
    For axis="col": draws vertical line at c=c_min of bbox
    """
    validate_colors(grid)
    h, w = dims(grid)

    if not (0 <= color <= 9):
        raise ValueError(f"Invalid color {color}; must be 0-9")

    if axis not in ["row", "col"]:
        raise ValueError(f"Invalid axis '{axis}'; must be 'row' or 'col'")

    result = deepcopy_grid(grid)
    r_min, r_max, c_min, c_max = get_mask_bbox(mask)

    if r_max < r_min:  # Empty mask
        return result

    if axis == "row":
        # Draw horizontal line at r_min (only on masked cells)
        for c in range(w):
            if mask[r_min][c]:
                result[r_min][c] = color
    else:  # axis == "col"
        # Draw vertical line at c_min (only on masked cells)
        for r in range(h):
            if mask[r][c_min]:
                result[r][c_min] = color

    return result
