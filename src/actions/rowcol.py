"""
Row/column reordering actions.

All operations are deterministic and only affect rows/columns
that intersect the mask.
"""

from typing import List
from src.utils import validate_colors, dims, deepcopy_grid
from src.actions import Grid, Mask


CONTRACT = {
    "reorder_rows_by_blocks": {"inputs": ["grid", "mask", "row_blocks"], "outputs": "grid", "mask_only": True,
                                "preconditions": ["colors in 0..9"]},
    "reorder_cols_by_blocks": {"inputs": ["grid", "mask", "col_blocks"], "outputs": "grid", "mask_only": True,
                                "preconditions": ["colors in 0..9"]},
    "sort_rows_lex": {"inputs": ["grid", "mask"], "outputs": "grid", "mask_only": True},
    "sort_cols_lex": {"inputs": ["grid", "mask"], "outputs": "grid", "mask_only": True},
}


def reorder_rows_by_blocks(grid: Grid, mask: Mask, row_blocks: List[List[int]]) -> Grid:
    """
    Reorder rows within each equivalence block.

    row_blocks: list of lists of row indices (equivalence classes from Φ)
    Within each block, sort rows by lexicographic order of masked cell values.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)

    for block in row_blocks:
        if len(block) <= 1:
            continue

        # Extract masked portions of rows in this block
        row_keys = []
        for r in block:
            key = tuple(grid[r][c] if mask[r][c] else -1 for c in range(w))
            row_keys.append((key, r))

        # Sort by key
        row_keys.sort()

        # Reorder: sorted_rows[i] goes to block[i]
        sorted_rows = [r for _, r in row_keys]
        for i, target_r in enumerate(block):
            source_r = sorted_rows[i]
            result[target_r] = grid[source_r][:]

    return result


def reorder_cols_by_blocks(grid: Grid, mask: Mask, col_blocks: List[List[int]]) -> Grid:
    """
    Reorder columns within each equivalence block.

    col_blocks: list of lists of column indices (equivalence classes from Φ)
    Within each block, sort columns by lexicographic order of masked cell values.
    """
    validate_colors(grid)
    h, w = dims(grid)

    result = deepcopy_grid(grid)

    for block in col_blocks:
        if len(block) <= 1:
            continue

        # Extract masked portions of columns in this block
        col_keys = []
        for c in block:
            key = tuple(grid[r][c] if mask[r][c] else -1 for r in range(h))
            col_keys.append((key, c))

        # Sort by key
        col_keys.sort()

        # Reorder: sorted_cols[i] goes to block[i]
        sorted_cols = [c for _, c in col_keys]
        for i, target_c in enumerate(block):
            source_c = sorted_cols[i]
            for r in range(h):
                result[r][target_c] = grid[r][source_c]

    return result


def sort_rows_lex(grid: Grid, mask: Mask) -> Grid:
    """
    Sort all rows lexicographically based on masked cell values.

    Rows are compared only on masked cells. Non-masked cells use -1 as placeholder.
    """
    validate_colors(grid)
    h, w = dims(grid)

    # Build row keys
    row_keys = []
    for r in range(h):
        key = tuple(grid[r][c] if mask[r][c] else -1 for c in range(w))
        row_keys.append((key, r))

    # Sort
    row_keys.sort()

    # Build result with sorted rows
    result = []
    for _, r in row_keys:
        result.append(grid[r][:])

    return result


def sort_cols_lex(grid: Grid, mask: Mask) -> Grid:
    """
    Sort all columns lexicographically based on masked cell values.

    Columns are compared only on masked cells. Non-masked cells use -1 as placeholder.
    """
    validate_colors(grid)
    h, w = dims(grid)

    # Build column keys
    col_keys = []
    for c in range(w):
        key = tuple(grid[r][c] if mask[r][c] else -1 for r in range(h))
        col_keys.append((key, c))

    # Sort
    col_keys.sort()

    # Build result with sorted columns
    result = [[0] * w for _ in range(h)]
    for new_c, (_, old_c) in enumerate(col_keys):
        for r in range(h):
            result[r][new_c] = grid[r][old_c]

    return result
