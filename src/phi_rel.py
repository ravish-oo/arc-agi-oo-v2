"""
Φ relational structure builder (input-only, no absolute coordinates in labels).

Builds a multi-relational graph from an input grid X:
  - Colors as unary atoms
  - E4: 4-adjacency
  - R_row: same-row equivalence
  - R_col: same-column equivalence
  - R_cc_k: same 4-connected component equivalence per color k (star topology)

Pure input-only; never accesses Y or Δ.
"""

from typing import List, Dict, Tuple
from collections import deque

Grid = List[List[int]]
Idx = int
Shape = Tuple[int, int]
RelationTag = str  # "E4", "R_row", "R_col", "R_cc_k" where k is color


class RelStructure:
    """
    Multi-relational graph over grid pixels.

    Flattened row-major indexing: i = r*w + c for grid of shape (h, w).
    """
    def __init__(self, h: int, w: int, n: int, colors: List[int],
                 neigh: Dict[RelationTag, List[List[int]]]):
        self.h = h
        self.w = w
        self.n = n
        self.colors = colors
        self.neigh = neigh


def build_rel_structure(X: Grid) -> RelStructure:
    """
    Build input-only multi-relational graph from grid X.

    Relations:
    - E4: 4-adjacency (up, down, left, right)
    - R_row: all pixels in same row
    - R_col: all pixels in same column
    - R_cc_k: star topology to deterministic representative per connected component of color k

    Deterministic; no absolute indices leaked to labels (only used to define relations).
    """
    if not X or not X[0]:
        # Empty grid
        return RelStructure(h=0, w=0, n=0, colors=[], neigh={})

    h = len(X)
    w = len(X[0])
    n = h * w

    # Flatten grid to colors array (row-major)
    colors = []
    for r in range(h):
        for c in range(w):
            colors.append(X[r][c])

    # Helper: convert (r, c) to flat index
    def idx(r: int, c: int) -> int:
        return r * w + c

    # Helper: convert flat index to (r, c)
    def rc(i: int) -> Tuple[int, int]:
        return (i // w, i % w)

    # Initialize neighbor lists
    neigh: Dict[RelationTag, List[List[int]]] = {}

    # 1. E4: 4-adjacency
    neigh["E4"] = [[] for _ in range(n)]
    for r in range(h):
        for c in range(w):
            i = idx(r, c)
            # Up
            if r > 0:
                neigh["E4"][i].append(idx(r - 1, c))
            # Down
            if r < h - 1:
                neigh["E4"][i].append(idx(r + 1, c))
            # Left
            if c > 0:
                neigh["E4"][i].append(idx(r, c - 1))
            # Right
            if c < w - 1:
                neigh["E4"][i].append(idx(r, c + 1))

    # 2. R_row: same row equivalence (each pixel connected to all others in same row)
    neigh["R_row"] = [[] for _ in range(n)]
    for r in range(h):
        row_indices = [idx(r, c) for c in range(w)]
        for i in row_indices:
            # Connect to all others in same row
            neigh["R_row"][i] = [j for j in row_indices if j != i]

    # 3. R_col: same column equivalence (each pixel connected to all others in same col)
    neigh["R_col"] = [[] for _ in range(n)]
    for c in range(w):
        col_indices = [idx(r, c) for r in range(h)]
        for i in col_indices:
            # Connect to all others in same column
            neigh["R_col"][i] = [j for j in col_indices if j != i]

    # 4. R_cc_k: connected component equivalence per color k (star topology)
    # Find all colors present
    unique_colors = sorted(set(colors))

    for k in unique_colors:
        tag = f"R_cc_{k}"
        neigh[tag] = [[] for _ in range(n)]

        # Find all pixels of color k
        k_pixels = [i for i in range(n) if colors[i] == k]

        if not k_pixels:
            continue

        # Find connected components using E4 within color k
        visited = set()
        components = []

        for start_i in k_pixels:
            if start_i in visited:
                continue

            # BFS to find component
            component = []
            queue = deque([start_i])
            visited.add(start_i)

            while queue:
                i = queue.popleft()
                component.append(i)

                # Check E4 neighbors that are also color k
                for j in neigh["E4"][i]:
                    if colors[j] == k and j not in visited:
                        visited.add(j)
                        queue.append(j)

            components.append(component)

        # For each component, choose representative as top-left (min row, then min col)
        # and create star edges
        for component in components:
            # Find top-left: min row, break ties by min col
            rep = min(component, key=lambda i: rc(i))

            # Star edges: rep <-> all nodes in component
            for i in component:
                if i != rep:
                    # Edge from i to rep
                    neigh[tag][i].append(rep)
                    # Edge from rep to i
                    neigh[tag][rep].append(i)

    return RelStructure(h=h, w=w, n=n, colors=colors, neigh=neigh)
