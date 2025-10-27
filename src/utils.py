"""
Utility functions for grid manipulation.
Pure functions, no external dependencies.
"""

from typing import List, Tuple


Grid = List[List[int]]


def grid_to_string(grid: Grid) -> str:
    """
    Convert grid to row-major string representation.
    Each cell is a digit 0-9.
    """
    return ''.join(str(cell) for row in grid for cell in row)


def same_grid(a: Grid, b: Grid) -> bool:
    """
    Check if two grids are identical (element-wise equality).
    """
    if len(a) != len(b):
        return False
    if not a:
        return True
    if len(a[0]) != len(b[0]):
        return False

    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] != b[i][j]:
                return False
    return True


def dims(grid: Grid) -> Tuple[int, int]:
    """
    Return (height, width) of grid.
    """
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]))


def deepcopy_grid(grid: Grid) -> Grid:
    """
    Create a deep copy of a grid.
    """
    return [row[:] for row in grid]


def validate_colors(grid: Grid) -> None:
    """
    Validate that all colors in grid are in range 0-9.
    Raises ValueError if any color is out of range.
    """
    h, w = dims(grid)
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if not (0 <= color <= 9):
                raise ValueError(f"Invalid color {color} at ({r},{c}); must be 0-9")
