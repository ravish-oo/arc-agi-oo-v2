"""
Structural learners for FY anti-overfit upgrade.

All learners are equality-only: they must find parameters that work consistently
across all training pairs or return None. No heuristics, no scores.
"""

from typing import List, Tuple, Optional, Dict
from collections import Counter
from src.utils import dims

Grid = List[List[int]]
Mask = List[List[bool]]


def fit_shift(trains: List[Tuple[Grid, Grid]], class_masks: List[Mask]) -> Optional[Tuple[int, int]]:
    """
    Learn (dr, dc) constant across pairs that maps masked X to Y on evidence.

    Returns (dr, dc) if same shift works for all pairs on evidence, else None.
    """
    if not trains:
        return None

    candidates = set()

    # First pair: collect all possible shifts from evidence
    Xp, Yp = trains[0]
    mask = class_masks[0]
    h, w = dims(Xp)

    for r in range(h):
        for c in range(w):
            if mask[r][c] and Xp[r][c] != Yp[r][c]:
                # This is evidence - find shift that would map Xp to Yp
                src_color = Xp[r][c]
                tgt_color = Yp[r][c]

                # Try shifts in range [-2, 2]
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        sr, sc = r - dr, c - dc
                        if 0 <= sr < h and 0 <= sc < w:
                            if Xp[sr][sc] == tgt_color:
                                candidates.add((dr, dc))

    if not candidates:
        return None

    # Test each candidate on all pairs
    for dr, dc in candidates:
        if all(_shift_matches_evidence(Xp, Yp, mask, dr, dc)
               for Xp, Yp, mask in zip([X for X, _ in trains],
                                       [Y for _, Y in trains],
                                       class_masks)):
            return (dr, dc)

    return None


def _shift_matches_evidence(Xp: Grid, Yp: Grid, mask: Mask, dr: int, dc: int) -> bool:
    """Check if shift (dr, dc) matches Y on all evidence pixels."""
    h, w = dims(Xp)

    for r in range(h):
        for c in range(w):
            if mask[r][c] and Xp[r][c] != Yp[r][c]:
                # Evidence pixel - check if shift produces correct value
                sr, sc = r - dr, c - dc
                if 0 <= sr < h and 0 <= sc < w:
                    if Xp[sr][sc] != Yp[r][c]:
                        return False
                else:
                    # Shift goes out of bounds - can't match
                    return False

    return True


def apply_shift(X: Grid, M: Mask, dr: int, dc: int) -> Grid:
    """
    Apply shift (dr, dc) inside mask only.

    Shifted-out cells are dropped, vacated masked cells become 0.
    """
    h, w = dims(X)
    result = [row[:] for row in X]

    for r in range(h):
        for c in range(w):
            if M[r][c]:
                sr, sc = r - dr, c - dc
                if 0 <= sr < h and 0 <= sc < w:
                    result[r][c] = X[sr][sc]
                else:
                    result[r][c] = 0

    return result


def fit_mirror(trains: List[Tuple[Grid, Grid]], class_masks: List[Mask]) -> Optional[str]:
    """
    Try mirror kinds {"H", "V", "Diag"}; accept if same works across all pairs on evidence.

    Returns kind ("H", "V", "Diag") or None.
    """
    if not trains:
        return None

    for kind in ["H", "V", "Diag"]:
        if all(_mirror_matches_evidence(Xp, Yp, mask, kind)
               for Xp, Yp, mask in zip([X for X, _ in trains],
                                       [Y for _, Y in trains],
                                       class_masks)):
            return kind

    return None


def _mirror_matches_evidence(Xp: Grid, Yp: Grid, mask: Mask, kind: str) -> bool:
    """Check if mirror kind matches Y on all evidence pixels."""
    h, w = dims(Xp)

    for r in range(h):
        for c in range(w):
            if mask[r][c] and Xp[r][c] != Yp[r][c]:
                # Evidence pixel - check if mirror produces correct value
                if kind == "H":
                    mr, mc = r, w - 1 - c
                elif kind == "V":
                    mr, mc = h - 1 - r, c
                elif kind == "Diag":
                    if h != w:
                        return False  # Diag requires square
                    mr, mc = c, r
                else:
                    return False

                if not (0 <= mr < h and 0 <= mc < w and mask[mr][mc]):
                    return False

                if Xp[mr][mc] != Yp[r][c]:
                    return False

    return True


def apply_mirror(X: Grid, M: Mask, kind: str) -> Grid:
    """
    Apply mirror inside mask only.

    Diag requires square bbox.
    """
    h, w = dims(X)
    result = [row[:] for row in X]

    for r in range(h):
        for c in range(w):
            if M[r][c]:
                if kind == "H":
                    mr, mc = r, w - 1 - c
                elif kind == "V":
                    mr, mc = h - 1 - r, c
                elif kind == "Diag":
                    if h != w:
                        continue  # Skip if not square
                    mr, mc = c, r
                else:
                    continue

                if 0 <= mr < h and 0 <= mc < w and M[mr][mc]:
                    result[r][c] = X[mr][mc]

    return result


def fit_row_perm(trains: List[Tuple[Grid, Grid]],
                 labels_per_pair: List[List[int]],
                 class_masks: List[Mask]) -> Optional[List[int]]:
    """
    Build row signatures from WL labels; find single permutation π that matches across all pairs.

    Returns permutation (list of row indices) or None.
    """
    if not trains:
        return None

    # For each pair, build row signatures
    pair_row_sigs = []
    for (Xp, Yp), labels, mask in zip(trains, labels_per_pair, class_masks):
        h, w = dims(Xp)

        # Row signature: multiset of WL labels in that row (masked only)
        x_sigs = []
        y_sigs = []
        for r in range(h):
            x_sig = tuple(sorted([labels[r * w + c] for c in range(w) if mask[r][c]]))
            # For Y, need to extract labels if they change
            y_colors = tuple(sorted([Yp[r][c] for c in range(w) if mask[r][c]]))
            x_sigs.append(x_sig)
            y_sigs.append(y_colors)

        pair_row_sigs.append((x_sigs, y_sigs))

    # Try to find a permutation that works for all pairs
    h = len(pair_row_sigs[0][0])

    # First pair determines the permutation
    x_sigs_0, y_sigs_0 = pair_row_sigs[0]

    # Try all permutations (brute force for small h)
    if h > 10:
        return None  # Too many permutations

    import itertools
    for perm in itertools.permutations(range(h)):
        if all(_perm_matches_pair(perm, x_sigs, y_sigs)
               for x_sigs, y_sigs in pair_row_sigs):
            return list(perm)

    return None


def _perm_matches_pair(perm: Tuple[int, ...], x_sigs: List, y_sigs: List) -> bool:
    """Check if permutation matches for this pair."""
    for i, pi in enumerate(perm):
        if x_sigs[pi] != y_sigs[i]:
            return False
    return True


def apply_row_perm(X: Grid, M: Mask, perm: List[int]) -> Grid:
    """Reorder masked rows only; outside-mask cells unchanged."""
    h, w = dims(X)
    result = [row[:] for row in X]

    # Apply permutation to rows
    for r in range(min(h, len(perm))):
        pr = perm[r]
        if pr < h:
            for c in range(w):
                if M[r][c]:
                    result[r][c] = X[pr][c]

    return result


def fit_col_perm(trains: List[Tuple[Grid, Grid]],
                 labels_per_pair: List[List[int]],
                 class_masks: List[Mask]) -> Optional[List[int]]:
    """Similar to fit_row_perm but for columns."""
    if not trains:
        return None

    # For each pair, build col signatures
    pair_col_sigs = []
    for (Xp, Yp), labels, mask in zip(trains, labels_per_pair, class_masks):
        h, w = dims(Xp)

        x_sigs = []
        y_sigs = []
        for c in range(w):
            x_sig = tuple(sorted([labels[r * w + c] for r in range(h) if mask[r][c]]))
            y_colors = tuple(sorted([Yp[r][c] for r in range(h) if mask[r][c]]))
            x_sigs.append(x_sig)
            y_sigs.append(y_colors)

        pair_col_sigs.append((x_sigs, y_sigs))

    w = len(pair_col_sigs[0][0])

    if w > 10:
        return None

    import itertools
    for perm in itertools.permutations(range(w)):
        if all(_perm_matches_pair(perm, x_sigs, y_sigs)
               for x_sigs, y_sigs in pair_col_sigs):
            return list(perm)

    return None


def apply_col_perm(X: Grid, M: Mask, perm: List[int]) -> Grid:
    """Reorder masked cols only; outside-mask cells unchanged."""
    h, w = dims(X)
    result = [row[:] for row in X]

    for c in range(min(w, len(perm))):
        pc = perm[c]
        if pc < w:
            for r in range(h):
                if M[r][c]:
                    result[r][c] = X[r][pc]

    return result


def fit_component_translate(trains: List[Tuple[Grid, Grid]],
                            class_masks: List[Mask]) -> Optional[Tuple[int, int]]:
    """
    For each component under the mask, infer one (dr, dc) that holds across pairs.

    Simplified: try to find a single (dr, dc) for all components.
    """
    # Similar to fit_shift but for components
    return fit_shift(trains, class_masks)


def apply_component_translate(X: Grid, M: Mask, dr: int, dc: int) -> Grid:
    """Copy component colors by (dr, dc) inside mask; clipped cells dropped."""
    return apply_shift(X, M, dr, dc)


def fit_block_substitution(trains: List[Tuple[Grid, Grid]],
                           class_masks: List[Mask],
                           k: int) -> Optional[Dict[int, Grid]]:
    """
    Each source color -> fixed k×k tile; must hold across pairs.

    Returns mapping {color: k×k_tile} or None.
    """
    if not trains or k not in {2, 3}:
        return None

    # Collect substitutions from all pairs
    subs = {}

    for (Xp, Yp), mask in zip(trains, class_masks):
        h, w = dims(Xp)

        # For each k×k block in mask
        for r in range(0, h - k + 1, k):
            for c in range(0, w - k + 1, k):
                # Check if block is fully masked
                if not all(mask[r + dr][c + dc] for dr in range(k) for dc in range(k)):
                    continue

                # Get source color (should be uniform in block)
                src_colors = {Xp[r + dr][c + dc] for dr in range(k) for dc in range(k)}
                if len(src_colors) != 1:
                    continue

                src_color = src_colors.pop()

                # Get target k×k tile
                tile = [[Yp[r + dr][c + dc] for dc in range(k)] for dr in range(k)]

                # Check consistency
                if src_color in subs:
                    if subs[src_color] != tile:
                        return None  # Conflict
                else:
                    subs[src_color] = tile

    return subs if subs else None


def apply_block_substitution(X: Grid, M: Mask, k: int, tiles: Dict[int, Grid]) -> Grid:
    """Apply block substitution inside mask."""
    h, w = dims(X)
    result = [row[:] for row in X]

    for r in range(0, h - k + 1, k):
        for c in range(0, w - k + 1, k):
            if not all(M[r + dr][c + dc] for dr in range(k) for dc in range(k)):
                continue

            # Get source color
            src_color = X[r][c]

            if src_color in tiles:
                tile = tiles[src_color]
                for dr in range(k):
                    for dc in range(k):
                        result[r + dr][c + dc] = tile[dr][dc]

    return result


def fit_block_permutation(trains: List[Tuple[Grid, Grid]],
                         class_masks: List[Mask],
                         k: int) -> Optional[List[int]]:
    """
    Partition into k×k tiles; learn global permutation consistent across pairs.

    Returns permutation of tile indices or None.
    """
    # Simplified: not implementing full block permutation for now
    return None


def apply_block_permutation(X: Grid, M: Mask, k: int, perm: List[int]) -> Grid:
    """Apply block permutation inside mask."""
    # Simplified: not implementing for now
    return X
