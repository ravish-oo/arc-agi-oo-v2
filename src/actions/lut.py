"""
LUT-r action: local pattern rewrite with radius r.

Extracts (2r+1)×(2r+1) patches, applies local OFA canonicalization,
looks up in dictionary, and replaces center pixel if key found.
"""

from typing import List, Dict, Tuple
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import local_ofa, Grid, Mask


CONTRACT = {
    "lut_rewrite": {"inputs": ["grid", "mask", "r", "key_to_color"], "outputs": "grid", "mask_only": True,
                    "preconditions": ["colors in 0..9", "r in {2,3}", "unseen_key->leave_unchanged"],
                    "policies": ["zero-pad", "deterministic"]},
}


def lut_rewrite(grid: Grid, mask: Mask, r: int,
                key_to_color: Dict[Tuple[Tuple[int, ...], ...], int]) -> Grid:
    """
    Apply local pattern rewrite using lookup table.

    Args:
        grid: input grid
        mask: boolean mask (only masked pixels are rewritten)
        r: patch radius (must be 2 or 3)
        key_to_color: dictionary mapping canonical patches to output colors

    Returns:
        New grid with masked pixels potentially rewritten

    Algorithm:
        1. For each masked pixel (r, c):
        2. Extract (2r+1)×(2r+1) patch centered at (r, c) with zero-padding
        3. Apply local OFA (first-appearance color relabeling) to make key
        4. If key in dictionary: replace pixel with mapped color
        5. If key not in dictionary: leave pixel unchanged (no guessing)
    """
    validate_colors(grid)
    h, w = dims(grid)

    if r not in {2, 3}:
        raise ValueError(f"Invalid radius {r}; must be 2 or 3")

    # Validate all colors in key_to_color values
    for color in key_to_color.values():
        if not (0 <= color <= 9):
            raise ValueError(f"Invalid color {color} in key_to_color; must be 0-9")

    result = deepcopy_grid(grid)

    # Process each masked pixel
    for row in range(h):
        for col in range(w):
            if not mask[row][col]:
                continue

            # Extract patch with zero-padding
            patch = []
            for dr in range(-r, r + 1):
                patch_row = []
                for dc in range(-r, r + 1):
                    pr = row + dr
                    pc = col + dc
                    if 0 <= pr < h and 0 <= pc < w:
                        patch_row.append(grid[pr][pc])
                    else:
                        # Zero-pad outside bounds
                        patch_row.append(0)
                patch.append(patch_row)

            # Apply local OFA to create canonical key
            key = local_ofa(patch)

            # Lookup and replace if found
            if key in key_to_color:
                result[row][col] = key_to_color[key]
            # else: leave unchanged (decline on unseen key)

    return result
