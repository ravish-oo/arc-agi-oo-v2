"""
M6 SOLVE — Full training equality + MDL + test prediction pipeline.

solve_task(trains, tests) -> SolveResult
  1) Π: canon_orient
  2) Enumerate Θ via M2
  3) Learn all exact candidates via M5
  4) If none exact -> UNSAT
  5) MDL pick among exacts
  6) Predict tests: Π → P → Φ → GLUE → undo Π
  7) Return ok=True with predictions
"""

from typing import List, Tuple, Dict, NamedTuple, Optional
from src.pi_orient import canon_orient, apply_transform, Canon
from src.p_menu import enumerate_feasible_P, apply_theta, Theta
from src.fy import learn_rules_via_wl_and_actions, LearnResult, Rulebook, Rule, mk_rowcol_blocks, AuxData
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.glue import glue_once
from src.mdl import Candidate, mdl_pick
from src.utils import dims

Grid = List[List[int]]


class SolveResult(NamedTuple):
    """Result of solve_task."""
    ok: bool
    theta: Optional[Theta]
    rulebook: Optional[Rulebook]
    preds: Optional[List[Grid]]  # test predictions if ok
    witness: Optional[Dict[str, object]]  # from M5 on UNSAT


def theta_to_p_family(theta: Optional[Theta]) -> str:
    """Convert theta to p_family string for MDL."""
    if theta is None:
        return "identity"
    return theta.kind


def rulebook_to_tuple_format(rulebook: Rulebook) -> Dict[int, Tuple[str, dict]]:
    """Convert Rulebook (Dict[int, Rule]) to Dict[int, (action_name, params)]."""
    result = {}
    for class_id, rule in rulebook.items():
        result[class_id] = (rule.action, rule.params)
    return result


def count_lut_keys(rulebook: Rulebook) -> int:
    """Count total LUT keys across all classes."""
    total = 0
    for rule in rulebook.values():
        if rule.action in ["lut_r2", "lut_r3"]:
            lut = rule.params.get("lut", {})
            total += len(lut)
    return total


def solve_task(
    trains: List[Tuple[Grid, Grid]],
    tests: List[Grid],
    escalate_policy: Optional[str] = None,
    use_task_color_canon: bool = False,
    lut_density_tau: float = 0.8
) -> SolveResult:
    """
    Full solve pipeline: Π → enumerate Θ → learn all exacts → MDL → predict tests.

    Args:
        trains: List of (input, output) training pairs
        tests: List of test input grids
        escalate_policy: Optional escalation policy (None, "E8", "2WL")
        use_task_color_canon: Whether to use task-level color canonicalization
        lut_density_tau: LUT density threshold for Gate D

    Returns:
        SolveResult with ok=True and predictions, or ok=False with witness
    """
    # Step 1: Π - canonize orientation
    trains_canon = []
    train_undos = []
    for X, Y in trains:
        X_canon = canon_orient(X)
        Y_canon = canon_orient(Y)
        trains_canon.append((X_canon.grid, Y_canon.grid))
        train_undos.append((X_canon, Y_canon))

    tests_canon = []
    test_undos = []
    for X in tests:
        X_canon = canon_orient(X)
        tests_canon.append(X_canon.grid)
        test_undos.append(X_canon)

    # Step 2: Enumerate Θ via M2
    thetas = enumerate_feasible_P(trains_canon)

    if len(thetas) == 0:
        # No feasible theta
        return SolveResult(
            ok=False,
            theta=None,
            rulebook=None,
            preds=None,
            witness={"reason": "no_feasible_theta"}
        )

    # Step 3: Learn all exact candidates via M5
    exact_candidates = []
    first_witness = None

    for theta in thetas:
        # Call M5's learn function
        result = learn_rules_via_wl_and_actions(
            trains_canon,
            [theta],  # Pass as list
            escalate_policy=escalate_policy,
            use_task_color_canon=use_task_color_canon,
            lut_density_tau=lut_density_tau
        )

        if result.ok:
            # This theta produced an exact match
            # Convert to Candidate format for MDL
            rulebook_tuple = rulebook_to_tuple_format(result.rulebook)
            num_classes = len(result.rulebook)
            lut_keys = count_lut_keys(result.rulebook)
            # Get actual escalation usage from LearnResult
            used_esc = result.used_escalation
            p_fam = theta_to_p_family(theta)

            cand = Candidate(
                theta=theta,
                rulebook=rulebook_tuple,
                num_classes_used=num_classes,
                lut_total_keys=lut_keys,
                used_escalation=used_esc,
                p_family=p_fam
            )
            exact_candidates.append((result, cand))
        else:
            # Keep first witness for reporting
            if first_witness is None:
                first_witness = result.witness

    # Step 4: If no exact candidates -> UNSAT
    if len(exact_candidates) == 0:
        return SolveResult(
            ok=False,
            theta=None,
            rulebook=None,
            preds=None,
            witness=first_witness
        )

    # Step 5: MDL pick among exact candidates
    candidates_only = [cand for _, cand in exact_candidates]
    chosen_cand = mdl_pick(candidates_only)

    # Find the corresponding LearnResult
    chosen_result = None
    for result, cand in exact_candidates:
        if cand == chosen_cand:
            chosen_result = result
            break

    assert chosen_result is not None, "MDL-chosen candidate must have a LearnResult"

    theta_star = chosen_result.theta
    rulebook_star = chosen_result.rulebook

    # Step 6: Predict tests
    preds = []
    for X_canon in tests_canon:
        # Apply P
        Xp = apply_theta(X_canon, theta_star)

        # Apply Φ (recompute with same settings as training)
        rel = build_rel_structure(Xp)
        labels, _, _ = wl_refine(rel, max_iters=20, escalate=escalate_policy)

        # Build aux (row/col blocks)
        h, w = dims(Xp)
        row_blocks, col_blocks = mk_rowcol_blocks(labels, h, w)
        aux = AuxData(row_blocks=row_blocks, col_blocks=col_blocks)

        # GLUE
        Yp = glue_once(Xp, labels, rulebook_star, aux)

        preds.append(Yp)

    # Step 7: Undo Π for test predictions
    preds_original = []
    for Yp, X_canon in zip(preds, test_undos):
        # Undo the orientation transformation applied to X
        Y_original = apply_transform(Yp, X_canon.undo_code)
        preds_original.append(Y_original)

    # Step 8: Return success
    return SolveResult(
        ok=True,
        theta=theta_star,
        rulebook=rulebook_star,
        preds=preds_original,
        witness=None
    )
