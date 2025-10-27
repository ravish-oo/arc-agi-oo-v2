"""
FY - Learning driver with anti-overfit gates (M5)

Implements Gates A-D to prevent overfitting:
- Gate A: Totality on mask
- Gate B: Non-evidence safety
- Gate C: Leave-one-out cross-validation
- Gate D: LUT regularity (key repetition, density, no collisions)
"""

from typing import List, Dict, Tuple, Optional, Literal, NamedTuple, Set
from collections import Counter, defaultdict
from src.utils import dims, deepcopy_grid, validate_colors, same_grid
from src.actions import local_ofa
from src.actions.mirror import mirror_h, mirror_v, mirror_diag
from src.actions.shift import shift
from src.actions.rowcol import reorder_rows_by_blocks, reorder_cols_by_blocks, sort_rows_lex, sort_cols_lex
from src.actions.constructors import draw_box_on_components, draw_line_axis_aligned
from src.actions.lut import lut_rewrite
from src.actions.constant import set_color
from src.pi_orient import canon_orient, apply_transform
from src.p_menu import Theta, apply_theta
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine


Grid = List[List[int]]
Mask = List[List[bool]]

# Global debug flag for per-class failure logging
FY_DEBUG_LOG = False
FY_DEBUG_TASK_ID = None
FY_DEBUG_THETA_KIND = None


class Rule(NamedTuple):
    """Action specification for a Φ-class."""
    action: Literal[
        "mirror_h", "mirror_v", "mirror_diag",
        "shift",
        "reorder_rows_by_blocks", "reorder_cols_by_blocks",
        "sort_rows_lex", "sort_cols_lex",
        "draw_box_on_components", "draw_line_axis_aligned",
        "lut_r2", "lut_r3",
        "set_color"
    ]
    params: Dict[str, object]


Rulebook = Dict[int, Rule]  # class_id -> Rule


class LearnResult(NamedTuple):
    """Result of learning attempt."""
    ok: bool
    theta: Optional[object]
    rulebook: Optional[Rulebook]
    witness: Optional[Dict[str, object]]
    used_escalation: bool


class AuxData(NamedTuple):
    """Auxiliary data for action application."""
    row_blocks: List[List[int]]
    col_blocks: List[List[int]]


# ============================================================================
# Helpers
# ============================================================================

def class_masks(labels: List[int], h: int, w: int) -> Dict[int, Mask]:
    """Build mask for each Φ-class."""
    masks = {}
    for k in set(labels):
        mask = [[False] * w for _ in range(h)]
        for idx, label in enumerate(labels):
            if label == k:
                r = idx // w
                c = idx % w
                mask[r][c] = True
        masks[k] = mask
    return masks


def mk_rowcol_blocks(labels: List[int], h: int, w: int) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Build row and column equivalence blocks based on label signatures.

    Rows with identical label signatures are grouped together.
    Same for columns.
    """
    # Build row signatures
    row_sigs = {}
    for r in range(h):
        sig = tuple(labels[r * w + c] for c in range(w))
        if sig not in row_sigs:
            row_sigs[sig] = []
        row_sigs[sig].append(r)
    row_blocks = [sorted(indices) for indices in row_sigs.values()]

    # Build col signatures
    col_sigs = {}
    for c in range(w):
        sig = tuple(labels[r * w + c] for r in range(h))
        if sig not in col_sigs:
            col_sigs[sig] = []
        col_sigs[sig].append(c)
    col_blocks = [sorted(indices) for indices in col_sigs.values()]

    return row_blocks, col_blocks


def apply_action(grid: Grid, mask: Mask, rule: Rule, aux: AuxData) -> Grid:
    """Apply a single action to a grid within a mask."""
    action = rule.action
    params = rule.params

    # Mirrors
    if action == "mirror_h":
        return mirror_h(grid, mask)
    elif action == "mirror_v":
        return mirror_v(grid, mask)
    elif action == "mirror_diag":
        return mirror_diag(grid, mask)

    # Shift
    elif action == "shift":
        dr = params["dr"]
        dc = params["dc"]
        return shift(grid, mask, dr, dc)

    # Row/col reordering
    elif action == "reorder_rows_by_blocks":
        return reorder_rows_by_blocks(grid, mask, aux.row_blocks)
    elif action == "reorder_cols_by_blocks":
        return reorder_cols_by_blocks(grid, mask, aux.col_blocks)
    elif action == "sort_rows_lex":
        return sort_rows_lex(grid, mask)
    elif action == "sort_cols_lex":
        return sort_cols_lex(grid, mask)

    # Constructors
    elif action == "draw_box_on_components":
        thickness = params.get("thickness", 1)
        color = params.get("color", None)
        return draw_box_on_components(grid, mask, thickness, color)
    elif action == "draw_line_axis_aligned":
        axis = params["axis"]
        color = params["color"]
        return draw_line_axis_aligned(grid, mask, axis, color)

    # LUT
    elif action in ["lut_r2", "lut_r3"]:
        lut = params["lut"]
        r = {"lut_r2": 2, "lut_r3": 3}[action]
        return lut_rewrite(grid, mask, r, lut)

    # Constant
    elif action == "set_color":
        color = params["color"]
        return set_color(grid, mask, color)

    else:
        raise ValueError(f"Unknown action: {action}")


def glue_once(Xp: Grid, labels: List[int], rulebook: Rulebook, aux: AuxData) -> Grid:
    """
    Compose all class edits on a grid.

    Per GLUE formula: Ŷ' = X' ⊕ ⊕_k (A_k(X') ⊙ M_k(X'))
    Each action A_k is applied to the ORIGINAL X', then masked.
    """
    h, w = dims(Xp)
    result = deepcopy_grid(Xp)
    masks = class_masks(labels, h, w)

    # Apply each action to ORIGINAL Xp, then copy masked pixels to result
    for class_id in sorted(rulebook.keys()):
        if class_id not in masks:
            continue
        rule = rulebook[class_id]
        mask = masks[class_id]

        # Apply action to ORIGINAL Xp (not accumulated result)
        edited = apply_action(Xp, mask, rule, aux)

        # Copy masked pixels from edited to result
        for r in range(h):
            for c in range(w):
                if mask[r][c]:
                    result[r][c] = edited[r][c]

    return result


# ============================================================================
# Action Simplicity Ranking
# ============================================================================

def action_simplicity_rank(rule: Rule) -> Tuple[int, int, int]:
    """
    Return (rank, param_bits, lut_size) for simplicity ordering.

    Lower is simpler. Ranks:
    0: mirrors, shifts
    1: row/col reorder/sort
    2: constructors
    3: LUT
    4: set_color
    """
    action = rule.action

    # Rank
    if action in ["mirror_h", "mirror_v", "mirror_diag", "shift"]:
        rank = 0
    elif action in ["reorder_rows_by_blocks", "reorder_cols_by_blocks", "sort_rows_lex", "sort_cols_lex"]:
        rank = 1
    elif action in ["draw_box_on_components", "draw_line_axis_aligned"]:
        rank = 2
    elif action in ["lut_r2", "lut_r3"]:
        rank = 3
    elif action == "set_color":
        rank = 4
    else:
        rank = 5

    # Param bits (rough estimate)
    param_bits = 0
    for k, v in rule.params.items():
        if isinstance(v, int):
            param_bits += 5  # Rough estimate
        elif isinstance(v, dict):
            param_bits += len(v) * 10  # LUT size

    # LUT size
    lut_size = 0
    if action in ["lut_r2", "lut_r3"] and "lut" in rule.params:
        lut_size = len(rule.params["lut"])

    return (rank, param_bits, lut_size)


# ============================================================================
# Gate Implementations
# ============================================================================

def check_gate_a_totality(
    rule: Rule,
    class_id: int,
    Xp_grids: List[Grid],
    labels_list: List[List[int]],
    aux_list: List[AuxData]
) -> bool:
    """
    Gate A: Totality on mask.

    For every pair and every pixel where M_k[p] is True,
    the action must define a value.

    For LUT: must cover 100% of masked pixels on training.
    """
    # For non-LUT actions, we assume they always define values
    # (mirrors, shifts, sorts, constructors, set_color all produce full grids)
    if rule.action not in ["lut_r2", "lut_r3"]:
        return True

    # For LUT: check coverage
    lut = rule.params.get("lut", {})
    if not lut:
        return False

    r = {"lut_r2": 2, "lut_r3": 3}[rule.action]

    for pair_idx in range(len(Xp_grids)):
        Xp = Xp_grids[pair_idx]
        labels = labels_list[pair_idx]
        h, w = dims(Xp)

        masks = class_masks(labels, h, w)
        if class_id not in masks:
            continue
        mask = masks[class_id]

        # Check every masked pixel has a LUT key
        for row in range(h):
            for col in range(w):
                if not mask[row][col]:
                    continue

                # Extract patch
                patch = []
                for dr in range(-r, r + 1):
                    patch_row = []
                    for dc in range(-r, r + 1):
                        pr = row + dr
                        pc = col + dc
                        if 0 <= pr < h and 0 <= pc < w:
                            patch_row.append(Xp[pr][pc])
                        else:
                            patch_row.append(0)
                    patch.append(patch_row)

                key = local_ofa(patch)
                if key not in lut:
                    return False  # Missing key - totality violated

    return True


def check_gate_b_non_evidence_safety(
    rule: Rule,
    class_id: int,
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    aux_list: List[AuxData]
) -> bool:
    """
    Gate B: Evidence matching AND non-evidence safety.

    For every pair and masked pixel:
    - If evidence (Xp[p] != Yp[p]): action must produce Yp[p]
    - If non-evidence (Xp[p] == Yp[p]): action must leave p unchanged
    """
    for pair_idx in range(len(Xp_grids)):
        Xp = Xp_grids[pair_idx]
        Yp = Yp_grids[pair_idx]
        labels = labels_list[pair_idx]
        aux = aux_list[pair_idx]
        h, w = dims(Xp)

        masks = class_masks(labels, h, w)
        if class_id not in masks:
            continue
        mask = masks[class_id]

        # Apply action
        try:
            result = apply_action(Xp, mask, rule, aux)
        except (ValueError, KeyError):
            return False

        # Check ALL masked pixels
        for r in range(h):
            for c in range(w):
                if mask[r][c]:
                    # BOTH evidence and non-evidence must match Yp
                    if result[r][c] != Yp[r][c]:
                        return False

    return True


def check_gate_c_leave_one_out(
    rule: Rule,
    class_id: int,
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    aux_list: List[AuxData]
) -> bool:
    """
    Gate C: Leave-one-out cross-validation.

    For each held-out pair h:
    - Learn action parameters/LUT on the other m-1 pairs
    - Apply to pair h
    - Require exact agreement on all evidence pixels of class k in h

    Only applies when m >= 2.
    """
    m = len(Xp_grids)
    if m < 2:
        # Leave-one-out doesn't apply for single training pair
        return True

    # For non-LUT actions, parameters are fixed (no learning)
    # Gate B already verified evidence matching on ALL pairs
    # So LOO is automatically satisfied (same action applies to all)
    if rule.action not in ["lut_r2", "lut_r3"]:
        return True

    # For LUT: do proper leave-one-out
    r = {"lut_r2": 2, "lut_r3": 3}[rule.action]

    for holdout_idx in range(m):
        # Build LUT from other pairs
        train_indices = [i for i in range(m) if i != holdout_idx]

        # Gather evidence from training pairs
        evidence_pixels = []
        for pair_idx in train_indices:
            Xp = Xp_grids[pair_idx]
            Yp = Yp_grids[pair_idx]
            labels = labels_list[pair_idx]
            h, w = dims(Xp)

            masks = class_masks(labels, h, w)
            if class_id not in masks:
                continue
            mask = masks[class_id]

            for row in range(h):
                for col in range(w):
                    if mask[row][col] and Xp[row][col] != Yp[row][col]:
                        evidence_pixels.append((pair_idx, row, col))

        if not evidence_pixels:
            continue

        # Build LUT from training pairs only
        lut_loo = build_lut_from_evidence(
            evidence_pixels, Xp_grids, Yp_grids, labels_list, class_id, r
        )

        if lut_loo is None:
            return False  # Collision in training

        # Apply to holdout pair
        Xp_h = Xp_grids[holdout_idx]
        Yp_h = Yp_grids[holdout_idx]
        labels_h = labels_list[holdout_idx]
        aux_h = aux_list[holdout_idx]
        h, w = dims(Xp_h)

        masks_h = class_masks(labels_h, h, w)
        if class_id not in masks_h:
            continue
        mask_h = masks_h[class_id]

        # Apply LUT action
        rule_loo = Rule(action=rule.action, params={"lut": lut_loo})
        try:
            result_h = apply_action(Xp_h, mask_h, rule_loo, aux_h)
        except (ValueError, KeyError):
            return False

        # Check evidence pixels in holdout
        for row in range(h):
            for col in range(w):
                if mask_h[row][col] and Xp_h[row][col] != Yp_h[row][col]:
                    if result_h[row][col] != Yp_h[row][col]:
                        return False  # LOO mismatch

    return True


def check_gate_d_lut_regularity(
    rule: Rule,
    class_id: int,
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    lut_density_tau: float = 0.8
) -> bool:
    """
    Gate D: LUT regularity.

    - No key collisions across training pairs (checked during build)
    - Repetition: with m>=2, every accepted LUT key must occur in >=2 distinct pairs
    - Density: LUT must cover >=tau of changed pixels per pair
    """
    if rule.action not in ["lut_r2", "lut_r3"]:
        return True

    lut = rule.params.get("lut", {})
    if not lut:
        return False

    r = {"lut_r2": 2, "lut_r3": 3}[rule.action]
    m = len(Xp_grids)

    # Check key repetition (m >= 2)
    if m >= 2:
        key_pair_counts = defaultdict(set)  # key -> set of pair indices

        for pair_idx in range(m):
            Xp = Xp_grids[pair_idx]
            labels = labels_list[pair_idx]
            h, w = dims(Xp)

            masks = class_masks(labels, h, w)
            if class_id not in masks:
                continue
            mask = masks[class_id]

            # Find keys in this pair
            for row in range(h):
                for col in range(w):
                    if not mask[row][col]:
                        continue

                    # Extract patch
                    patch = []
                    for dr in range(-r, r + 1):
                        patch_row = []
                        for dc in range(-r, r + 1):
                            pr = row + dr
                            pc = col + dc
                            if 0 <= pr < h and 0 <= pc < w:
                                patch_row.append(Xp[pr][pc])
                            else:
                                patch_row.append(0)
                        patch.append(patch_row)

                    key = local_ofa(patch)
                    if key in lut:
                        key_pair_counts[key].add(pair_idx)

        # Check every key appears in >= 2 pairs
        for key, pair_set in key_pair_counts.items():
            if len(pair_set) < 2:
                return False  # Key only in 1 pair - overfitting

    # Check density per pair
    for pair_idx in range(m):
        Xp = Xp_grids[pair_idx]
        Yp = Yp_grids[pair_idx]
        labels = labels_list[pair_idx]
        h, w = dims(Xp)

        masks = class_masks(labels, h, w)
        if class_id not in masks:
            continue
        mask = masks[class_id]

        # Count changed pixels in class
        changed_count = 0
        covered_count = 0

        for row in range(h):
            for col in range(w):
                if not mask[row][col]:
                    continue
                if Xp[row][col] != Yp[row][col]:
                    changed_count += 1

                    # Check if LUT covers this pixel
                    patch = []
                    for dr in range(-r, r + 1):
                        patch_row = []
                        for dc in range(-r, r + 1):
                            pr = row + dr
                            pc = col + dc
                            if 0 <= pr < h and 0 <= pc < w:
                                patch_row.append(Xp[pr][pc])
                            else:
                                patch_row.append(0)
                        patch.append(patch_row)

                    key = local_ofa(patch)
                    if key in lut:
                        covered_count += 1

        if changed_count > 0:
            density = covered_count / changed_count
            if density < lut_density_tau:
                return False  # Density too low

    return True


def build_lut_from_evidence(
    evidence_pairs: List[Tuple[int, int, int]],  # (pair_idx, r, c)
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    class_id: int,
    r: int
) -> Optional[Dict[Tuple[Tuple[int, ...], ...], int]]:
    """Build LUT from training evidence. Returns None on collision."""
    if r not in {2, 3}:
        raise ValueError(f"Invalid LUT radius {r}")

    lut = {}

    for pair_idx, row, col in evidence_pairs:
        Xp = Xp_grids[pair_idx]
        Yp = Yp_grids[pair_idx]
        h, w = dims(Xp)

        # Extract patch with zero-padding
        patch = []
        for dr in range(-r, r + 1):
            patch_row = []
            for dc in range(-r, r + 1):
                pr = row + dr
                pc = col + dc
                if 0 <= pr < h and 0 <= pc < w:
                    patch_row.append(Xp[pr][pc])
                else:
                    patch_row.append(0)
            patch.append(patch_row)

        key = local_ofa(patch)
        target_color = Yp[row][col]

        # Check for collision
        if key in lut:
            if lut[key] != target_color:
                return None  # Collision
        else:
            lut[key] = target_color

    return lut


# ============================================================================
# Main Learning Function
# ============================================================================

def print_class_diagnostics(class_id, evidence, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau, theta):
    """Print detailed diagnostics for why a class failed."""
    if not FY_DEBUG_LOG:
        return

    # Compute masks for this class (one per pair)
    masks = []
    for pair_idx, (Xp, labels) in enumerate(zip(Xp_grids, labels_list)):
        h, w = dims(Xp)
        all_masks = class_masks(labels, h, w)
        mask = all_masks.get(class_id, [[False] * w for _ in range(h)])
        masks.append(mask)

    mask_size = sum(sum(1 for cell in row if cell) for mask in masks for row in mask)
    evidence_size = len(evidence)

    print(f"\n=== FY CLASS FAILURE DIAGNOSTIC ===")
    print(f"class_id: {class_id}")
    print(f"mask_size: {mask_size}")
    print(f"evidence_count: {evidence_size}")
    print(f"theta: {theta.kind if theta else 'identity'}")
    print(f"\nACTIONS_TRIED:")

    # Get evidence per pair
    evidence_by_pair = defaultdict(list)
    for pair_idx, r, c in evidence:
        evidence_by_pair[pair_idx].append((r, c))

    m = len(Xp_grids)

    # Try mirrors
    for mirror_kind, mirror_name in [("h", "mirror_h"), ("v", "mirror_v"), ("diag", "mirror_diag")]:
        # Check totality (Gate A)
        masks_flat = []
        for mask in masks:
            for row in mask:
                masks_flat.extend(row)
        totality_ok = all(masks_flat)

        # Check safety (Gate B) - would this break non-evidence pixels?
        safety_ok = True
        for i, (Xp, Yp) in enumerate(zip(Xp_grids, Yp_grids)):
            mask = masks[i]
            if mirror_kind == "h":
                edited = mirror_h(Xp, mask)
            elif mirror_kind == "v":
                edited = mirror_v(Xp, mask)
            else:
                edited = mirror_diag(Xp, mask)

            # Check non-evidence pixels
            h, w = dims(Xp)
            for r in range(h):
                for c in range(w):
                    if mask[r][c] and (i, r, c) not in [(pi, pr, pc) for pi, pr, pc in evidence]:
                        # Non-evidence pixel in mask
                        if Xp[r][c] == Yp[r][c]:  # Should remain unchanged
                            if edited[r][c] != Yp[r][c]:
                                safety_ok = False
                                break
                if not safety_ok:
                    break
            if not safety_ok:
                break

        # Check LOO (Gate C) if m >= 2
        loo_ok = True
        if m >= 2:
            for leave_out in range(m):
                # Try learning on all pairs except leave_out
                train_pairs = [(Xp_grids[i], Yp_grids[i]) for i in range(m) if i != leave_out]
                # For mirrors, just check if it matches the test pair
                test_Xp = Xp_grids[leave_out]
                test_Yp = Yp_grids[leave_out]
                test_mask = masks[leave_out]

                if mirror_kind == "h":
                    test_edited = mirror_h(test_Xp, test_mask)
                elif mirror_kind == "v":
                    test_edited = mirror_v(test_Xp, test_mask)
                else:
                    test_edited = mirror_diag(test_Xp, test_mask)

                # Check if evidence matches
                for r, c in evidence_by_pair[leave_out]:
                    if test_edited[r][c] != test_Yp[r][c]:
                        loo_ok = False
                        break
                if not loo_ok:
                    break

        result = "PASS" if (totality_ok and safety_ok and loo_ok) else "FAIL"
        print(f"  - {mirror_name}: totality={totality_ok}, safety={safety_ok}, loo={loo_ok} -> {result}")

    # Try shifts
    print(f"  - shift: (trying dr,dc in [-2,2])")
    shift_found = False
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            # Quick check if this shift works
            works = True
            for i, (Xp, Yp) in enumerate(zip(Xp_grids, Yp_grids)):
                mask = masks[i]
                edited = shift(Xp, mask, dr, dc)
                # Check evidence
                for r, c in evidence_by_pair[i]:
                    if edited[r][c] != Yp[r][c]:
                        works = False
                        break
                if not works:
                    break
            if works:
                print(f"    shift({dr},{dc}): PASS")
                shift_found = True
                break
        if shift_found:
            break
    if not shift_found:
        print(f"    no shift found: FAIL")

    # Try LUT
    for r in [2, 3]:
        # Build LUT from evidence
        lut = {}
        collisions = 0
        for i, (Xp, Yp) in enumerate(zip(Xp_grids, Yp_grids)):
            mask = masks[i]
            labels = labels_list[i]
            h, w = dims(Xp)

            for r_pos in range(h):
                for c_pos in range(w):
                    if not mask[r_pos][c_pos]:
                        continue

                    # Extract r×r patch
                    if r_pos + r > h or c_pos + r > w:
                        continue

                    patch = tuple(tuple(Xp[r_pos+dr][c_pos+dc] for dc in range(r)) for dr in range(r))
                    target = Yp[r_pos][c_pos]

                    if patch in lut:
                        if lut[patch] != target:
                            collisions += 1
                    else:
                        lut[patch] = target

        repetition_ok = len([1 for patch in lut.keys() if sum(1 for i in range(len(Xp_grids)) for r_pos in range(len(Xp_grids[i])) for c_pos in range(len(Xp_grids[i][0])) if masks[i][r_pos][c_pos] and r_pos+r<=len(Xp_grids[i]) and c_pos+r<=len(Xp_grids[i][0]) and tuple(tuple(Xp_grids[i][r_pos+dr][c_pos+dc] for dc in range(r)) for dr in range(r)) == patch) >= 2]) >= 2

        coverage_changed = sum(1 for i in range(len(Xp_grids)) for r_pos, c_pos in evidence_by_pair[i])
        density = len(lut) / coverage_changed if coverage_changed > 0 else 0

        result = "PASS" if (collisions == 0 and repetition_ok and density <= lut_density_tau) else "FAIL"
        print(f"  - lut_r{r}: |lut|={len(lut)}, collisions={collisions}, repetition_ok={repetition_ok}, density={density:.2f} -> {result}")

    print(f"===\n")


def learn_rules_via_wl_and_actions(
    trains: List[Tuple[Grid, Grid]],
    thetas: List[object],
    escalate_policy: Optional[Literal["E8", "2WL"]] = None,
    use_task_color_canon: bool = False,
    lut_density_tau: float = 0.8
) -> LearnResult:
    """
    Learn rulebook from training pairs.

    Try each theta in order. For the first theta whose per-class actions
    pass all gates and yield exact equality on all training pairs,
    return ok=True with its rulebook.

    Otherwise return ok=False with a concise witness.
    """
    if not trains:
        return LearnResult(ok=False, theta=None, rulebook=None,
                          witness={"reason": "no_training_data"},
                          used_escalation=False)

    # If no thetas provided, try with theta=None (identity)
    if not thetas:
        thetas = [None]

    # Try each theta
    for theta in thetas:
        result = learn_for_one_theta(
            trains, theta, escalate_policy,
            use_task_color_canon, lut_density_tau
        )
        if result.ok:
            return result

    # All thetas failed
    return LearnResult(ok=False, theta=thetas[0] if thetas else None,
                      rulebook=None,
                      witness={"reason": "all_thetas_failed"},
                      used_escalation=False)


def debug_glue_mismatch(Xp, Yp, labels, rulebook, aux, r, c, failing_class, pair_idx):
    """
    Detailed GLUE mismatch diagnostics for checks A, B, C, D.
    """
    print(f"\n  === GLUE DEBUG for class {failing_class} at ({r},{c}) ===")

    h, w = dims(Xp)
    idx = r * w + c

    # A) Mask & labels consistency check
    print(f"\n  [A] Mask & Labels Consistency:")
    print(f"    labels[{r},{c}] = {labels[idx]} (should be {failing_class})")

    # Build mask for failing class
    all_masks = class_masks(labels, h, w)
    mask = all_masks.get(failing_class, [[False] * w for _ in range(h)])
    print(f"    mask[{r},{c}] = {mask[r][c]}")
    print(f"    Xp[{r},{c}] = {Xp[r][c]}")
    print(f"    Yp[{r},{c}] = {Yp[r][c]}")

    # B) Action safety recheck for this class
    print(f"\n  [B] Action Safety Recheck for class {failing_class}:")
    if failing_class in rulebook:
        rule = rulebook[failing_class]
        print(f"    Action: {rule.action}")
        print(f"    Params: {rule.params}")

        # Apply this action only
        edited = apply_action(Xp, mask, rule, aux)

        # Count violations
        changed_clean = 0
        wrong_evidence = 0
        for rr in range(h):
            for cc in range(w):
                if mask[rr][cc]:
                    if Xp[rr][cc] == Yp[rr][cc]:  # Non-evidence (clean)
                        if edited[rr][cc] != Xp[rr][cc]:
                            changed_clean += 1
                    else:  # Evidence
                        if edited[rr][cc] != Yp[rr][cc]:
                            wrong_evidence += 1

        print(f"    changed_clean = {changed_clean} (should be 0)")
        print(f"    wrong_evidence = {wrong_evidence} (should be 0)")
        print(f"    edited[{r},{c}] = {edited[r][c]} (vs Yp={Yp[r][c]})")
    else:
        print(f"    WARNING: class {failing_class} NOT in rulebook!")

    # C) Incremental GLUE class-by-class
    print(f"\n  [C] Incremental GLUE (class-by-class):")
    Z = deepcopy_grid(Xp)
    first_bad_class = None
    all_masks = class_masks(labels, h, w)

    for class_id in sorted(rulebook.keys()):
        rule = rulebook[class_id]
        mask_k = all_masks.get(class_id, [[False] * w for _ in range(h)])

        # Apply action for this class
        edited_k = apply_action(Xp, mask_k, rule, aux)

        # Merge into Z
        for rr in range(h):
            for cc in range(w):
                if mask_k[rr][cc]:
                    Z[rr][cc] = edited_k[rr][cc]

        # Check mismatches after this class
        mismatches_now = sum(1 for rr in range(h) for cc in range(w) if Z[rr][cc] != Yp[rr][cc])

        if mismatches_now > 0 and first_bad_class is None:
            first_bad_class = class_id
            print(f"    First bad class: {class_id} ({rule.action})")
            print(f"      Mismatches after applying: {mismatches_now}")

            if class_id == failing_class:
                print(f"      Z[{r},{c}] = {Z[r][c]} (after class {class_id})")
            break

    if first_bad_class is None:
        print(f"    No bad class found (all incremental steps matched!)")

    # D) Canon toggle consistency (N/A for now - we're not using task color canon)
    print(f"\n  [D] Canon Toggle:")
    print(f"    use_task_color_canon = False (not implemented)")
    print(f"  === END GLUE DEBUG ===\n")


def learn_for_one_theta(
    trains: List[Tuple[Grid, Grid]],
    theta: Optional[object],
    escalate_policy: Optional[Literal["E8", "2WL"]],
    use_task_color_canon: bool,
    lut_density_tau: float
) -> LearnResult:
    """Learn rulebook for a specific theta."""

    # Debug: announce which theta we're trying
    if FY_DEBUG_LOG:
        print(f"\n{'='*70}")
        print(f"TRYING THETA: {theta.kind if theta else 'identity'}")
        print(f"{'='*70}")

    # Step 1: Canon and apply P
    trains_transformed = []
    for X, Y in trains:
        Xc = canon_orient(X).grid
        Yc = canon_orient(Y).grid
        if theta:
            Xp = apply_theta(Xc, theta)
        else:
            Xp = Xc
        Yp = Yc  # Never apply P to Y
        trains_transformed.append((Xp, Yp))

    # Optional task color canon (input-only) - NOT IMPLEMENTED
    # Would need to build color frequency map from training Xp only

    # Step 2: Φ per transformed input
    labels_list = []
    for Xp, _ in trains_transformed:
        rel = build_rel_structure(Xp)
        labels, _, _ = wl_refine(rel, max_iters=20, escalate=None)
        labels_list.append(labels)

    # Step 3: Evidence and masks
    # Evidence: S_k = {(i,r,c) | Xp[i][r][c] != Yp[i][r][c] and labels[i][r*w+c] == k}
    evidence_sets = defaultdict(list)
    for pair_idx, (Xp, Yp) in enumerate(trains_transformed):
        labels = labels_list[pair_idx]
        h, w = dims(Xp)
        h_y, w_y = dims(Yp)

        # Dimension mismatch - theta produced different sized grids
        if h != h_y or w != w_y:
            if FY_DEBUG_LOG:
                print(f"  → DIMENSION_MISMATCH: Xp={h}×{w}, Yp={h_y}×{w_y} at pair {pair_idx}")
            return LearnResult(
                ok=False,
                theta=theta,
                rulebook={},
                witness={"reason": "dimension_mismatch", "pair_idx": pair_idx},
                used_escalation=False
            )

        for r in range(h):
            for c in range(w):
                idx = r * w + c
                if Xp[r][c] != Yp[r][c]:
                    class_id = labels[idx]
                    evidence_sets[class_id].append((pair_idx, r, c))

    # Debug: print evidence summary
    if FY_DEBUG_LOG:
        total_evidence = sum(len(v) for v in evidence_sets.values())
        print(f"  → Classes with evidence: {len(evidence_sets)}, Total evidence pixels: {total_evidence}")

    # Step 4: Row/col blocks
    aux_list = []
    for Xp, _ in trains_transformed:
        h, w = dims(Xp)
        idx = len(aux_list)
        row_blocks, col_blocks = mk_rowcol_blocks(labels_list[idx], h, w)
        aux_list.append(AuxData(row_blocks=row_blocks, col_blocks=col_blocks))

    # Step 5: Learn action for each class
    Xp_grids = [Xp for Xp, _ in trains_transformed]
    Yp_grids = [Yp for _, Yp in trains_transformed]

    rulebook = {}
    all_classes = sorted(set(k for k in evidence_sets.keys()))

    escalated = False  # Track if we've escalated once

    for class_id in all_classes:
        if class_id not in evidence_sets or not evidence_sets[class_id]:
            # No evidence - identity/no-op
            continue

        # Try to learn action for this class
        rule = try_learn_action_for_class_with_gates(
            class_id, evidence_sets[class_id],
            Xp_grids, Yp_grids, labels_list, aux_list,
            lut_density_tau
        )

        if rule is None:
            # Print diagnostics if debug enabled
            print_class_diagnostics(class_id, evidence_sets[class_id], Xp_grids, Yp_grids,
                                   labels_list, aux_list, lut_density_tau, theta)

            # No action passed gates - try escalation once per task
            if escalate_policy and not escalated:
                escalated = True

                # Recompute Φ with escalation for ALL pairs
                labels_list_esc = []
                for Xp, _ in trains_transformed:
                    rel = build_rel_structure(Xp)
                    labels_esc, _, _ = wl_refine(rel, max_iters=20, escalate=escalate_policy)
                    labels_list_esc.append(labels_esc)

                # Rebuild evidence
                evidence_sets_esc = defaultdict(list)
                for pair_idx, (Xp, Yp) in enumerate(trains_transformed):
                    labels_esc = labels_list_esc[pair_idx]
                    h, w = dims(Xp)
                    for r in range(h):
                        for c in range(w):
                            idx = r * w + c
                            if Xp[r][c] != Yp[r][c]:
                                class_id_esc = labels_esc[idx]
                                evidence_sets_esc[class_id_esc].append((pair_idx, r, c))

                # Rebuild aux
                aux_list_esc = []
                for Xp, _ in trains_transformed:
                    h, w = dims(Xp)
                    idx = len(aux_list_esc)
                    row_blocks, col_blocks = mk_rowcol_blocks(labels_list_esc[idx], h, w)
                    aux_list_esc.append(AuxData(row_blocks=row_blocks, col_blocks=col_blocks))

                # Retry learning with escalated Φ
                return learn_for_one_theta_after_escalation(
                    trains_transformed, theta,
                    labels_list_esc, evidence_sets_esc, aux_list_esc,
                    lut_density_tau
                )

            # No escalation or already escalated - UNSAT
            sample = evidence_sets[class_id][0] if evidence_sets[class_id] else (0, 0, 0)
            return LearnResult(
                ok=False, theta=theta, rulebook=None,
                witness={
                    "class_id": class_id,
                    "reason": "no_action_passes_gates",
                    "sample": sample
                },
                used_escalation=escalated
            )

        rulebook[class_id] = rule

    # Step 6: Glue check
    if FY_DEBUG_LOG:
        print(f"  → Learned rulebook with {len(rulebook)} classes, checking GLUE...")

    for i, (Xp, Yp) in enumerate(trains_transformed):
        labels = labels_list[i]
        aux = aux_list[i]
        composed = glue_once(Xp, labels, rulebook, aux)

        if not same_grid(composed, Yp):
            # Find mismatch
            h, w = dims(Xp)
            mismatch_count = 0
            first_mismatch = None
            for r in range(h):
                for c in range(w):
                    if composed[r][c] != Yp[r][c]:
                        mismatch_count += 1
                        if first_mismatch is None:
                            idx = r * w + c
                            class_at_pos = labels[idx]
                            first_mismatch = (r, c, Yp[r][c], composed[r][c], class_at_pos)

            if FY_DEBUG_LOG and first_mismatch:
                r, c, expected, got, cls = first_mismatch
                print(f"  → GLUE_MISMATCH at pair {i}: {mismatch_count} mismatches")
                print(f"     First at ({r},{c}): expected={expected}, got={got}, class={cls}")

                # Run detailed GLUE diagnostics
                debug_glue_mismatch(Xp, Yp, labels, rulebook, aux, r, c, cls, i)

            return LearnResult(
                ok=False, theta=theta, rulebook=None,
                witness={
                    "reason": "glue_mismatch",
                    "pair_idx": i,
                    "pos": (first_mismatch[0], first_mismatch[1]),
                    "expected": first_mismatch[2],
                    "got": first_mismatch[3]
                },
                used_escalation=escalated
            )

    # Success!
    return LearnResult(ok=True, theta=theta, rulebook=rulebook, witness=None,
                      used_escalation=escalated)


def learn_for_one_theta_after_escalation(
    trains_transformed: List[Tuple[Grid, Grid]],
    theta: Optional[object],
    labels_list_esc: List[List[int]],
    evidence_sets_esc: Dict[int, List[Tuple[int, int, int]]],
    aux_list_esc: List[AuxData],
    lut_density_tau: float
) -> LearnResult:
    """Learn after single escalation."""
    Xp_grids = [Xp for Xp, _ in trains_transformed]
    Yp_grids = [Yp for _, Yp in trains_transformed]

    rulebook = {}
    all_classes = sorted(set(k for k in evidence_sets_esc.keys()))

    for class_id in all_classes:
        if class_id not in evidence_sets_esc or not evidence_sets_esc[class_id]:
            continue

        rule = try_learn_action_for_class_with_gates(
            class_id, evidence_sets_esc[class_id],
            Xp_grids, Yp_grids, labels_list_esc, aux_list_esc,
            lut_density_tau
        )

        if rule is None:
            # Still failed after escalation
            sample = evidence_sets_esc[class_id][0] if evidence_sets_esc[class_id] else (0, 0, 0)
            return LearnResult(
                ok=False, theta=theta, rulebook=None,
                witness={
                    "class_id": class_id,
                    "reason": "no_action_passes_gates_after_escalation",
                    "sample": sample
                },
                used_escalation=True
            )

        rulebook[class_id] = rule

    # Glue check
    for i, (Xp, Yp) in enumerate(trains_transformed):
        labels = labels_list_esc[i]
        aux = aux_list_esc[i]
        composed = glue_once(Xp, labels, rulebook, aux)

        if not same_grid(composed, Yp):
            h, w = dims(Xp)
            for r in range(h):
                for c in range(w):
                    if composed[r][c] != Yp[r][c]:
                        return LearnResult(
                            ok=False, theta=theta, rulebook=None,
                            witness={
                                "reason": "glue_mismatch_after_escalation",
                                "pair_idx": i,
                                "pos": (r, c),
                                "expected": Yp[r][c],
                                "got": composed[r][c]
                            },
                            used_escalation=True
                        )

    return LearnResult(ok=True, theta=theta, rulebook=rulebook, witness=None,
                      used_escalation=True)


def try_learn_action_for_class_with_gates(
    class_id: int,
    evidence: List[Tuple[int, int, int]],
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    aux_list: List[AuxData],
    lut_density_tau: float
) -> Optional[Rule]:
    """
    Try action menu in fixed order.
    Return first action that passes all gates.
    """
    candidates = []

    # 1. Mirrors
    for action in ["mirror_h", "mirror_v", "mirror_diag"]:
        rule = Rule(action=action, params={})
        if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
            candidates.append(rule)

    # 2. Shifts
    for dr in [-2, -1, 0, 1, 2]:
        for dc in [-2, -1, 0, 1, 2]:
            if dr == 0 and dc == 0:
                continue
            rule = Rule(action="shift", params={"dr": dr, "dc": dc})
            if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
                candidates.append(rule)

    # 3. Row/col reorder/sort
    for action in ["reorder_rows_by_blocks", "reorder_cols_by_blocks", "sort_rows_lex", "sort_cols_lex"]:
        rule = Rule(action=action, params={})
        if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
            candidates.append(rule)

    # 4. Constructors
    # draw_box_on_components variations
    for thickness in [1, 2]:
        for color in list(range(10)) + [None]:
            rule = Rule(action="draw_box_on_components", params={"thickness": thickness, "color": color})
            if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
                candidates.append(rule)

    # draw_line_axis_aligned
    for axis in ["row", "col"]:
        for color in range(10):
            rule = Rule(action="draw_line_axis_aligned", params={"axis": axis, "color": color})
            if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
                candidates.append(rule)

    # 5. LUT (try r=2, r=3, optionally r=4)
    for r in [2, 3]:
        # Build LUT from all evidence
        lut = build_lut_from_evidence(evidence, Xp_grids, Yp_grids, labels_list, class_id, r)
        if lut is not None:
            action_name = f"lut_r{r}"
            rule = Rule(action=action_name, params={"lut": lut})
            if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
                candidates.append(rule)

    # 6. set_color
    for color in range(10):
        rule = Rule(action="set_color", params={"color": color})
        if passes_all_gates(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list, lut_density_tau):
            candidates.append(rule)

    # Return simplest candidate
    if not candidates:
        return None

    candidates.sort(key=action_simplicity_rank)
    return candidates[0]


def passes_all_gates(
    rule: Rule,
    class_id: int,
    Xp_grids: List[Grid],
    Yp_grids: List[Grid],
    labels_list: List[List[int]],
    aux_list: List[AuxData],
    lut_density_tau: float
) -> bool:
    """Check if rule passes Gates A, B, C, D.

    Ordered cheapest-first to fail fast:
    B (evidence only) → D (key counting) → A (totality) → C (LOO - most expensive)
    """

    # Gate B: Non-evidence safety (cheapest - only checks evidence pixels)
    if not check_gate_b_non_evidence_safety(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list):
        return False

    # Gate D: LUT regularity (moderate - scans for key repetition/density)
    if not check_gate_d_lut_regularity(rule, class_id, Xp_grids, Yp_grids, labels_list, lut_density_tau):
        return False

    # Gate A: Totality (moderate - checks all masked pixels have keys)
    if not check_gate_a_totality(rule, class_id, Xp_grids, labels_list, aux_list):
        return False

    # Gate C: Leave-one-out (most expensive - rebuilds LUT m times)
    if not check_gate_c_leave_one_out(rule, class_id, Xp_grids, Yp_grids, labels_list, aux_list):
        return False

    return True
