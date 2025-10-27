#!/usr/bin/env python3
"""
M5 Debug: Per-class failure logging for a single task.

Runs solve_task with detailed action failure logging.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable FY debug logging
import src.fy as fy
fy.FY_DEBUG_LOG = True

from src.solve import solve_task
from src.pi_orient import canon_orient
from src.p_menu import enumerate_feasible_P, apply_theta
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.fy import mk_rowcol_blocks, AuxData, class_masks
from src.utils import dims


def load_single_task(data_dir: Path, task_id: str):
    """Load a single task by ID."""
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    if task_id not in data:
        print(f"ERROR: Task {task_id} not found")
        sys.exit(1)

    task = data[task_id]
    train_pairs = task.get('train', [])
    trains = [(p['input'], p['output']) for p in train_pairs]
    tests = [p['input'] for p in task.get('test', [])]

    return trains, tests


def detailed_class_diagnostics(task_id, trains, theta):
    """
    Run detailed diagnostics on why each class fails for a given theta.

    Prints per-class, per-action failure reasons.
    """
    from src.fy import try_learn_action_for_class_with_gates
    from collections import defaultdict

    print(f"\nTASK {task_id}  THETA {theta.kind if theta else 'identity'}")
    print("=" * 70)

    # Transform pairs
    trains_transformed = []
    for X, Y in trains:
        Xc = canon_orient(X).grid
        Yc = canon_orient(Y).grid
        Xp = apply_theta(Xc, theta)
        trains_transformed.append((Xp, Yc))

    # Compute Φ for all pairs
    labels_list = []
    for Xp, _ in trains_transformed:
        rel = build_rel_structure(Xp)
        labels, _, _ = wl_refine(rel, max_iters=20)
        labels_list.append(labels)

    # Build evidence sets
    evidence_sets = defaultdict(list)
    for pair_idx, (Xp, Yp) in enumerate(trains_transformed):
        labels = labels_list[pair_idx]
        h, w = dims(Xp)
        for r in range(h):
            for c in range(w):
                idx = r * w + c
                if Xp[r][c] != Yp[r][c]:
                    class_id = labels[idx]
                    evidence_sets[class_id].append((pair_idx, r, c))

    # Build aux
    aux_list = []
    for Xp, _ in trains_transformed:
        h, w = dims(Xp)
        idx = len(aux_list)
        row_blocks, col_blocks = mk_rowcol_blocks(labels_list[idx], h, w)
        aux_list.append(AuxData(row_blocks=row_blocks, col_blocks=col_blocks))

    Xp_grids = [Xp for Xp, _ in trains_transformed]
    Yp_grids = [Yp for _, Yp in trains_transformed]

    all_classes = sorted(set(k for k in evidence_sets.keys()))

    print(f"Total classes with evidence: {len(all_classes)}")
    print()

    # For each class, try to learn and report failures
    for class_id in all_classes[:5]:  # Show first 5 classes
        if class_id not in evidence_sets or not evidence_sets[class_id]:
            continue

        evidence = evidence_sets[class_id]

        # Compute mask for this class
        masks_list = []
        for labels in labels_list:
            h = len(Xp_grids[len(masks_list)])
            w = len(Xp_grids[len(masks_list)][0]) if h > 0 else 0
            mask = [[labels[r * w + c] == class_id for c in range(w)] for r in range(h)]
            masks_list.append(mask)

        # Count mask size and evidence
        mask_size = sum(sum(1 for cell in row if cell) for mask in masks_list for row in mask)
        evidence_size = len(evidence)

        print(f"CLASS {class_id}:")
        print(f"  MASK_SIZE: {mask_size}")
        print(f"  EVIDENCE: {evidence_size}")

        # Try learn
        rule = try_learn_action_for_class_with_gates(
            class_id, evidence,
            Xp_grids, Yp_grids, labels_list, aux_list,
            lut_density_tau=0.8
        )

        if rule is None:
            print(f"  RESULT: ✗ All actions failed gates")
        else:
            print(f"  RESULT: ✓ {rule.action} with params {rule.params}")

        print()


def main():
    parser = argparse.ArgumentParser(description='M5 Debug: Single task per-class failure logging')
    parser.add_argument('--kaggle_path', type=str, default='data',
                       help='Path to ARC data directory')
    parser.add_argument('--task_id', type=str, required=True,
                       help='Single task ID to debug')

    args = parser.parse_args()

    data_dir = Path(args.kaggle_path)
    trains, tests = load_single_task(data_dir, args.task_id)

    print("=" * 70)
    print(f"M5 DEBUG: Single Task Analysis")
    print("=" * 70)
    print(f"Task ID: {args.task_id}")
    print(f"Train pairs: {len(trains)}")
    print(f"Test inputs: {len(tests)}")
    print()

    # Get feasible thetas
    trains_pi = []
    for X, Y in trains:
        Xc = canon_orient(X).grid
        Yc = canon_orient(Y).grid
        trains_pi.append((Xc, Yc))

    thetas = enumerate_feasible_P(trains_pi)
    print(f"Feasible Θ: {[t.kind for t in thetas]}")
    print()

    if not thetas:
        print("No feasible theta - cannot proceed with FY analysis")
        return 1

    # Try first theta
    theta = thetas[0]
    detailed_class_diagnostics(args.task_id, trains, theta)

    # Now run full solve_task
    print()
    print("=" * 70)
    print("FULL SOLVE_TASK RUN")
    print("=" * 70)

    fy.FY_DEBUG_TASK_ID = args.task_id
    result = solve_task(trains, tests, escalate_policy=None)

    print(f"\nResult: ok={result.ok}")
    if result.ok:
        print(f"Theta: {result.theta.kind if result.theta else '?'}")
        print(f"Rulebook size: {len(result.rulebook)}")
    else:
        print(f"UNSAT: {result.witness}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
