"""
P (global map) menu - feasibility-only, shape-based transform enumeration.

All functions are pure shape algebra. No pixel content is accessed.
Deterministic ordering: identity → isometries → transpose → scale_up → scale_down → tile_repeat.
"""

from typing import List, Tuple, Dict, NamedTuple, Literal, Optional
from collections import Counter

Grid = List[List[int]]
Shape = Tuple[int, int]  # (height, width)


class Theta(NamedTuple):
    """
    A global map parameterization.

    kind: transform type
    params: numeric parameters (e.g., scale factor)
    extra: string parameters as sorted tuple (e.g., (("agg", "majority"),))
    """
    kind: Literal[
        "identity", "FH", "FV", "R90", "R180", "R270", "transpose",
        "scale_up", "scale_down", "tile_repeat"
    ]
    params: Tuple[int, ...]
    extra: Tuple[Tuple[str, str], ...]  # Sorted tuple of (key, value) pairs


def output_shape_for_theta(in_shape: Shape, theta: Theta) -> Optional[Shape]:
    """
    Compute output shape for a given input shape and theta.
    Returns None if the transform is not defined for this shape.

    Pure shape mathematics only - no pixel access.
    """
    h, w = in_shape

    if theta.kind == "identity":
        return (h, w)

    elif theta.kind == "FH":
        # Flip horizontal: shape unchanged
        return (h, w)

    elif theta.kind == "FV":
        # Flip vertical: shape unchanged
        return (h, w)

    elif theta.kind == "R90":
        # Rotate 90° clockwise: transpose dimensions
        return (w, h)

    elif theta.kind == "R180":
        # Rotate 180°: shape unchanged
        return (h, w)

    elif theta.kind == "R270":
        # Rotate 270° clockwise: transpose dimensions
        return (w, h)

    elif theta.kind == "transpose":
        # Explicit transpose
        return (w, h)

    elif theta.kind == "scale_up":
        # Scale up by factor s
        s = theta.params[0]
        return (h * s, w * s)

    elif theta.kind == "scale_down":
        # Scale down by pooling blocks of size b×b
        b = theta.params[0]
        if h % b != 0 or w % b != 0:
            return None
        return (h // b, w // b)

    elif theta.kind == "tile_repeat":
        # Repeat grid by factors (kh, kw)
        kh, kw = theta.params
        return (h * kh, w * kw)

    else:
        raise ValueError(f"Unknown theta kind: {theta.kind}")


def enumerate_theta_candidates(in_shape: Shape, out_shape: Shape) -> List[Theta]:
    """
    Enumerate all theta that transform in_shape to out_shape.

    Deterministic order:
    1. identity
    2. isometries (FH, FV, R90, R180, R270)
    3. transpose
    4. scale_up (s=2, then s=3)
    5. scale_down (b=2: majority, first_nonzero; b=3: majority, first_nonzero, center)
    6. tile_repeat (sorted by (kh, kw))

    Only uses shapes - no pixel content.
    """
    candidates = []
    h_in, w_in = in_shape
    h_out, w_out = out_shape

    # 1. Identity
    if (h_in, w_in) == (h_out, w_out):
        candidates.append(Theta(kind="identity", params=(), extra=()))

    # 2. Isometries
    # FH, FV: shape unchanged
    if (h_in, w_in) == (h_out, w_out):
        candidates.append(Theta(kind="FH", params=(), extra=()))
        candidates.append(Theta(kind="FV", params=(), extra=()))

    # R90, R270: transpose dimensions
    if (w_in, h_in) == (h_out, w_out):
        candidates.append(Theta(kind="R90", params=(), extra=()))

    # R180: shape unchanged
    if (h_in, w_in) == (h_out, w_out):
        candidates.append(Theta(kind="R180", params=(), extra=()))

    if (w_in, h_in) == (h_out, w_out):
        candidates.append(Theta(kind="R270", params=(), extra=()))

    # 3. Transpose (explicit, for rectangles)
    if (w_in, h_in) == (h_out, w_out):
        candidates.append(Theta(kind="transpose", params=(), extra=()))

    # 4. Scale up
    for s in [2, 3]:
        if (h_in * s, w_in * s) == (h_out, w_out):
            candidates.append(Theta(kind="scale_up", params=(s,), extra=()))

    # 5. Scale down
    for b in [2, 3]:
        if h_in % b == 0 and w_in % b == 0:
            if (h_in // b, w_in // b) == (h_out, w_out):
                # Aggregators
                candidates.append(Theta(kind="scale_down", params=(b,), extra=(("agg", "majority"),)))
                candidates.append(Theta(kind="scale_down", params=(b,), extra=(("agg", "first_nonzero"),)))

                # Center only for b=3
                if b == 3:
                    candidates.append(Theta(kind="scale_down", params=(b,), extra=(("agg", "center"),)))

    # 6. Tile repeat
    # Check if output dimensions are multiples of input dimensions
    if h_out % h_in == 0 and w_out % w_in == 0:
        kh = h_out // h_in
        kw = w_out // w_in
        if kh >= 1 and kw >= 1:
            candidates.append(Theta(kind="tile_repeat", params=(kh, kw), extra=()))

    return candidates


def enumerate_feasible_P(trains_pi: List[Tuple[Grid, Grid]]) -> List[Theta]:
    """
    Given Π-canonized training pairs, return thetas feasible for ALL pairs.

    Feasibility = shape compatibility only (no content equality).
    Returns intersection of candidates across all pairs.
    Preserves deterministic ordering.
    """
    if not trains_pi:
        return []

    # Get candidates for first pair
    first_X, first_Y = trains_pi[0]
    h_X, w_X = len(first_X), len(first_X[0]) if first_X else 0
    h_Y, w_Y = len(first_Y), len(first_Y[0]) if first_Y else 0

    feasible_set = set(enumerate_theta_candidates((h_X, w_X), (h_Y, w_Y)))

    # Intersect with candidates from remaining pairs
    for X, Y in trains_pi[1:]:
        h_X, w_X = len(X), len(X[0]) if X else 0
        h_Y, w_Y = len(Y), len(Y[0]) if Y else 0

        pair_candidates = set(enumerate_theta_candidates((h_X, w_X), (h_Y, w_Y)))
        feasible_set = feasible_set.intersection(pair_candidates)

        if not feasible_set:
            break

    # Return in deterministic order by re-enumerating and filtering
    # Use first pair's shapes to get canonical order, then filter
    all_ordered = enumerate_theta_candidates((h_X, w_X), (h_Y, w_Y))

    # But we need to be careful - feasible_set might contain thetas not in all_ordered
    # if different pairs have different shapes
    # So we'll sort by a canonical key instead

    result = sorted(feasible_set, key=lambda t: _theta_sort_key(t))
    return result


def _theta_sort_key(theta: Theta) -> Tuple:
    """
    Generate a sort key for deterministic ordering of thetas.
    Order: identity → isometries → transpose → scale_up → scale_down → tile_repeat
    """
    kind_order = {
        "identity": 0,
        "FH": 1,
        "FV": 2,
        "R90": 3,
        "R180": 4,
        "R270": 5,
        "transpose": 6,
        "scale_up": 7,
        "scale_down": 8,
        "tile_repeat": 9,
    }

    # Primary key: kind order
    primary = kind_order.get(theta.kind, 999)

    # Secondary key: params (as tuple)
    secondary = theta.params

    # Tertiary key: extra (already sorted tuple)
    tertiary = theta.extra

    return (primary, secondary, tertiary)


def apply_theta(X: Grid, theta: Theta) -> Grid:
    """
    Apply a global transform theta to grid X.
    Returns a new grid (does not mutate input).

    Note: This is a helper for testing/verification.
    In production, P application happens in the main pipeline.
    """
    from src.pi_orient import apply_transform
    from src.utils import deepcopy_grid

    h, w = len(X), len(X[0]) if X else 0

    if theta.kind == "identity":
        return deepcopy_grid(X)

    elif theta.kind in ["FH", "FV", "R90", "R180", "R270"]:
        # Use pi_orient transforms
        return apply_transform(X, theta.kind)

    elif theta.kind == "transpose":
        # Use FMD (main diagonal flip) from pi_orient
        return apply_transform(X, "FMD")

    elif theta.kind == "scale_up":
        s = theta.params[0]
        result = [[0 for _ in range(w * s)] for _ in range(h * s)]
        for r in range(h):
            for c in range(w):
                color = X[r][c]
                # Replicate to s×s block
                for dr in range(s):
                    for dc in range(s):
                        result[r * s + dr][c * s + dc] = color
        return result

    elif theta.kind == "scale_down":
        b = theta.params[0]
        # Extract aggregator from extra tuple
        agg = dict(theta.extra).get("agg", "majority")

        h_out, w_out = h // b, w // b
        result = [[0 for _ in range(w_out)] for _ in range(h_out)]

        for r_out in range(h_out):
            for c_out in range(w_out):
                # Extract b×b block
                block = []
                for dr in range(b):
                    for dc in range(b):
                        block.append(X[r_out * b + dr][c_out * b + dc])

                # Apply aggregator
                if agg == "majority":
                    # Count colors, pick most common (ties broken by smallest color)
                    counts = Counter(block)
                    max_count = max(counts.values())
                    candidates = [color for color, count in counts.items() if count == max_count]
                    result[r_out][c_out] = min(candidates)

                elif agg == "first_nonzero":
                    # Scan row-major, pick first nonzero
                    first_nz = 0
                    for color in block:
                        if color != 0:
                            first_nz = color
                            break
                    result[r_out][c_out] = first_nz

                elif agg == "center":
                    # Center of 3×3 block is at (1, 1)
                    if b == 3:
                        result[r_out][c_out] = X[r_out * 3 + 1][c_out * 3 + 1]
                    else:
                        raise ValueError("center aggregator only valid for b=3")

        return result

    elif theta.kind == "tile_repeat":
        kh, kw = theta.params
        result = [[0 for _ in range(w * kw)] for _ in range(h * kh)]

        for r in range(h * kh):
            for c in range(w * kw):
                # Map to source tile
                src_r = r % h
                src_c = c % w
                result[r][c] = X[src_r][src_c]

        return result

    else:
        raise ValueError(f"Unknown theta kind: {theta.kind}")
