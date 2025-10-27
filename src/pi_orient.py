"""
Π (orientation canonicalization) implementation.

Tries a fixed list of grid isometries, picks the lexicographically smallest
row-major string, and returns the canonical image with an undo code.
"""

from typing import List
from src.utils import grid_to_string, deepcopy_grid, dims


Grid = List[List[int]]


# Fixed transform order (deterministic)
TRANSFORMS = [
    "I",            # identity
    "R90",          # rotate 90° clockwise
    "R180",         # rotate 180°
    "R270",         # rotate 270° clockwise
    "FH",           # flip horizontal (left-right mirror)
    "FV",           # flip vertical (top-bottom mirror)
    "FMD",          # flip main diagonal (transpose, y=x)
    "FAD",          # flip anti-diagonal (y=-x)
]


class Canon:
    """Result of canonicalization."""
    def __init__(self, grid: Grid, undo_code: str):
        self.grid = grid
        self.undo_code = undo_code


def apply_transform(Z: Grid, code: str) -> Grid:
    """
    Apply a single transform to grid Z.
    Returns a new grid (does not mutate input).
    """
    h, w = dims(Z)

    if code == "I":
        # Identity
        return deepcopy_grid(Z)

    elif code == "R90":
        # Rotate 90° clockwise: new grid is w×h
        # Z[r][c] -> result[c][h-1-r]
        result = [[0 for _ in range(h)] for _ in range(w)]
        for r in range(h):
            for c in range(w):
                result[c][h - 1 - r] = Z[r][c]
        return result

    elif code == "R180":
        # Rotate 180°: new grid is h×w
        # Z[r][c] -> result[h-1-r][w-1-c]
        result = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                result[h - 1 - r][w - 1 - c] = Z[r][c]
        return result

    elif code == "R270":
        # Rotate 270° clockwise: new grid is w×h
        # Z[r][c] -> result[w-1-c][r]
        result = [[0 for _ in range(h)] for _ in range(w)]
        for r in range(h):
            for c in range(w):
                result[w - 1 - c][r] = Z[r][c]
        return result

    elif code == "FH":
        # Flip horizontal (left-right mirror): new grid is h×w
        # Z[r][c] -> result[r][w-1-c]
        result = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                result[r][w - 1 - c] = Z[r][c]
        return result

    elif code == "FV":
        # Flip vertical (top-bottom mirror): new grid is h×w
        # Z[r][c] -> result[h-1-r][c]
        result = [[0 for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                result[h - 1 - r][c] = Z[r][c]
        return result

    elif code == "FMD":
        # Flip main diagonal (transpose): new grid is w×h
        # Z[r][c] -> result[c][r]
        result = [[0 for _ in range(h)] for _ in range(w)]
        for r in range(h):
            for c in range(w):
                result[c][r] = Z[r][c]
        return result

    elif code == "FAD":
        # Flip anti-diagonal: new grid is w×h
        # Z[r][c] -> result[w-1-c][h-1-r]
        result = [[0 for _ in range(h)] for _ in range(w)]
        for r in range(h):
            for c in range(w):
                result[w - 1 - c][h - 1 - r] = Z[r][c]
        return result

    else:
        raise ValueError(f"Unknown transform code: {code}")


def inverse(code: str) -> str:
    """
    Return the inverse transform code.

    Inverse mappings:
    - I ↔ I
    - R90 ↔ R270
    - R180 ↔ R180
    - FH ↔ FH
    - FV ↔ FV
    - FMD ↔ FMD
    - FAD ↔ FAD
    """
    inverses = {
        "I": "I",
        "R90": "R270",
        "R180": "R180",
        "R270": "R90",
        "FH": "FH",
        "FV": "FV",
        "FMD": "FMD",
        "FAD": "FAD",
    }
    return inverses[code]


def canon_orient(Z: Grid) -> Canon:
    """
    Canonicalize grid orientation by trying all transforms and picking
    the lexicographically smallest row-major string.

    Returns Canon with the canonical grid and the undo code to reverse it.
    Deterministic and idempotent.
    """
    best_str = None
    best_code = None
    best_grid = None

    for code in TRANSFORMS:
        Zt = apply_transform(Z, code)
        s = grid_to_string(Zt)

        if best_str is None or s < best_str:
            best_str = s
            best_code = code
            best_grid = Zt

    return Canon(grid=best_grid, undo_code=inverse(best_code))
