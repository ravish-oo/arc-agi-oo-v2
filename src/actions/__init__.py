"""
Common helpers for action implementations.

All actions operate on grids with boolean masks and must:
- Call validate_colors(grid) and dims(grid) at entry
- Never modify cells outside mask
- Return new grid (no mutation)
- Use zero-padding for neighborhoods
- Use deterministic row-major iteration
"""

from typing import List, Tuple
from collections import deque


Grid = List[List[int]]
Mask = List[List[bool]]


def get_mask_bbox(mask: Mask) -> Tuple[int, int, int, int]:
    """
    Get bounding box of mask.
    Returns (r_min, r_max, c_min, c_max) inclusive.
    If mask is empty, returns (0, -1, 0, -1).
    """
    if not mask or not mask[0]:
        return (0, -1, 0, -1)

    h = len(mask)
    w = len(mask[0])

    r_min, r_max = h, -1
    c_min, c_max = w, -1

    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)

    return (r_min, r_max, c_min, c_max)


def find_e4_components(mask: Mask) -> List[List[Tuple[int, int]]]:
    """
    Find 4-connected components in mask.
    Returns list of components, where each component is a list of (r, c) coordinates.
    """
    if not mask or not mask[0]:
        return []

    h = len(mask)
    w = len(mask[0])

    visited = [[False] * w for _ in range(h)]
    components = []

    for r in range(h):
        for c in range(w):
            if mask[r][c] and not visited[r][c]:
                # BFS to find component
                component = []
                queue = deque([(r, c)])
                visited[r][c] = True

                while queue:
                    cr, cc = queue.popleft()
                    component.append((cr, cc))

                    # Check 4 neighbors
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < h and 0 <= nc < w and
                            mask[nr][nc] and not visited[nr][nc]):
                            visited[nr][nc] = True
                            queue.append((nr, nc))

                components.append(component)

    return components


def local_ofa(patch: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """
    Apply local Order of First Appearance (OFA) to a patch.

    Returns canonical patch as tuple of tuples where colors are relabeled
    0, 1, 2, ... in order of first appearance (row-major scan).

    This makes patches palette-invariant for LUT keys.
    """
    if not patch or not patch[0]:
        return tuple()

    h = len(patch)
    w = len(patch[0])

    # Build first-appearance mapping
    first_appearance = {}
    next_id = 0

    canonical = []
    for r in range(h):
        row = []
        for c in range(w):
            color = patch[r][c]
            if color not in first_appearance:
                first_appearance[color] = next_id
                next_id += 1
            row.append(first_appearance[color])
        canonical.append(tuple(row))

    return tuple(canonical)
