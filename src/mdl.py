"""
M6 MDL — T6 deterministic tie-break among exact candidates.

Cost tuple (lexicographic min):
  (num_classes_used, action_complexity_sum, p_cost, lut_total_keys,
   used_escalation, action_name_lex_tuple)

Only operates on candidates that are bit-exact on training.
"""

from typing import List, Dict, Tuple, NamedTuple, Optional


class Candidate(NamedTuple):
    """An exact candidate (theta, rulebook) from learning."""
    theta: object  # from M2
    rulebook: Dict[int, Tuple[str, dict]]  # class_id -> (action_name, params)
    num_classes_used: int
    lut_total_keys: int
    used_escalation: bool
    p_family: str  # e.g., "identity", "FH", "scale_up", etc.


# Action complexity table: rank * 16 + bits
# Ranks: mirror/shift=0, reorder/sort=1, constructors=2, LUT=3, set_color=4

def action_rank(action_name: str) -> int:
    """Return the coarse rank for an action."""
    if action_name in ["mirror_h", "mirror_v", "mirror_diag", "shift"]:
        return 0
    elif action_name in ["reorder_rows_by_blocks", "reorder_cols_by_blocks",
                         "sort_rows_lex", "sort_cols_lex"]:
        return 1
    elif action_name in ["draw_box_on_components", "draw_line_axis_aligned"]:
        return 2
    elif action_name in ["lut_r2", "lut_r3"]:
        return 3
    elif action_name == "set_color":
        return 4
    else:
        return 5  # unknown


def action_bits(action_name: str, params: dict) -> int:
    """Return the fine-grained bit count for an action."""
    if action_name in ["mirror_h", "mirror_v", "mirror_diag"]:
        return 1
    elif action_name == "shift":
        # Encode (dr, dc) in [-2..2]^2 -> 0..24 (5x5 grid)
        dr = params.get("dr", 0)
        dc = params.get("dc", 0)
        # Map [-2,-1,0,1,2] -> [0,1,2,3,4]
        dr_idx = dr + 2
        dc_idx = dc + 2
        return 1 + dr_idx * 5 + dc_idx
    elif action_name in ["reorder_rows_by_blocks", "reorder_cols_by_blocks",
                         "sort_rows_lex", "sort_cols_lex"]:
        return 2
    elif action_name == "draw_box_on_components":
        # 3 + thickness_bit + (color_provided?1:0)
        thickness = params.get("thickness", 1)
        thickness_bit = 0 if thickness == 1 else 1
        color_provided = 1 if "color" in params else 0
        return 3 + thickness_bit + color_provided
    elif action_name == "draw_line_axis_aligned":
        # 3 + axis_bit + 1
        axis = params.get("axis", "row")
        axis_bit = 0 if axis == "row" else 1
        return 3 + axis_bit + 1
    elif action_name in ["lut_r2", "lut_r3"]:
        # 10 + #keys
        lut = params.get("lut", {})
        return 10 + len(lut)
    elif action_name == "set_color":
        return 1
    else:
        return 0


def action_complexity(action_name: str, params: dict) -> int:
    """Return rank * 16 + bits for an action."""
    rank = action_rank(action_name)
    bits = action_bits(action_name, params)
    return rank * 16 + bits


def p_family_cost(p_family: str) -> int:
    """Return the cost for a P family."""
    costs = {
        "identity": 0,
        "FH": 1, "FV": 1, "R180": 1,
        "R90": 2, "R270": 2,
        "transpose": 3,
        "scale_up": 4,
        "scale_down": 5,
        "tile_repeat": 6,
    }
    return costs.get(p_family, 7)  # unknown gets 7


def compute_cost_tuple(cand: Candidate) -> Tuple:
    """
    Compute the MDL cost tuple for a candidate.

    Returns: (num_classes_used, action_complexity_sum, p_cost, lut_total_keys,
              used_escalation_int, action_name_lex_tuple)
    """
    num_classes = cand.num_classes_used
    lut_keys = cand.lut_total_keys
    esc_int = 1 if cand.used_escalation else 0
    p_cost = p_family_cost(cand.p_family)

    # Action complexity sum
    action_sum = 0
    action_names = []
    for class_id in sorted(cand.rulebook.keys()):  # deterministic order
        action_name, params = cand.rulebook[class_id]
        action_sum += action_complexity(action_name, params)
        action_names.append(action_name)

    # Action names tuple (sorted for deterministic tie-breaking)
    action_name_tuple = tuple(sorted(action_names))

    return (num_classes, action_sum, p_cost, lut_keys, esc_int, action_name_tuple)


def mdl_pick(exacts: List[Candidate]) -> Candidate:
    """
    Return the lexicographically minimal candidate by MDL cost tuple.

    Precondition: exacts is non-empty and all candidates are bit-exact on training.
    """
    assert len(exacts) > 0, "mdl_pick requires at least one exact candidate"

    # Compute cost tuples and pair with candidates
    candidates_with_costs = [(compute_cost_tuple(cand), cand) for cand in exacts]

    # Sort by cost tuple (lexicographic order)
    candidates_with_costs.sort(key=lambda x: x[0])

    # Return the minimal candidate
    return candidates_with_costs[0][1]
