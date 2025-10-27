#!/usr/bin/env python3
"""
M2 Debug: Detailed Θ feasibility debugging

Prints per-pair candidates, intersection, and sanity checks for identity/transpose.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient
from src.p_menu import enumerate_feasible_P, Theta

Grid = List[List[int]]


def load_specific_tasks(data_dir: Path, task_ids: List[str]):
    """Load specific tasks by ID."""
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    tasks = {}
    for task_id in task_ids:
        if task_id in data:
            task = data[task_id]
            train_pairs = task.get('train', [])
            pairs = [(p['input'], p['output']) for p in train_pairs]
            tasks[task_id] = pairs
        else:
            print(f"WARNING: Task {task_id} not found")

    return tasks


def get_per_pair_candidates(trains_pi: List[Tuple[Grid, Grid]]) -> List[List[Theta]]:
    """
    Get feasible candidates for each pair individually (before intersection).

    Uses enumerate_theta_candidates from p_menu which is shape-based only.
    """
    from src.p_menu import enumerate_theta_candidates
    from src.utils import dims

    per_pair_candidates = []

    for Xc, Yc in trains_pi:
        h_in, w_in = dims(Xc)
        h_out, w_out = dims(Yc)

        # Get shape-based candidates for this pair
        candidates = enumerate_theta_candidates((h_in, w_in), (h_out, w_out))
        per_pair_candidates.append(candidates)

    return per_pair_candidates


def main():
    parser = argparse.ArgumentParser(description='M2 Debug: Θ feasibility analysis')
    parser.add_argument('--kaggle_path', type=str, required=True,
                       help='Path to ARC data directory')
    parser.add_argument('--task_ids', type=str, default=None,
                       help='Comma-separated task IDs to debug')
    parser.add_argument('--scan_first', type=int, default=None,
                       help='Scan first N tasks and print Θ_feas info')

    args = parser.parse_args()

    data_dir = Path(args.kaggle_path)

    # Handle scan mode
    if args.scan_first is not None:
        print("=" * 70)
        print(f"M2 DEBUG: Scanning first {args.scan_first} tasks for Θ_feas")
        print("=" * 70)
        print()

        # Load all training tasks
        challenges_path = data_dir / "arc-agi_training_challenges.json"
        if not challenges_path.exists():
            print(f"ERROR: {challenges_path} not found")
            sys.exit(1)

        with open(challenges_path, 'r') as f:
            data = json.load(f)

        count = 0
        for task_id in sorted(data.keys()):
            if count >= args.scan_first:
                break

            task = data[task_id]
            train_pairs = task.get('train', [])
            pairs = [(p['input'], p['output']) for p in train_pairs]

            # Canonize pairs
            trains_pi = []
            for X, Y in pairs:
                Xc = canon_orient(X).grid
                Yc = canon_orient(Y).grid
                trains_pi.append((Xc, Yc))

            # Get Θ_feas
            thetas_feas = enumerate_feasible_P(trains_pi)
            theta_kinds = [t.kind for t in thetas_feas]

            print(f"{task_id}: |Θ|={len(thetas_feas)} {theta_kinds}")

            count += 1

        print()
        print(f"Scanned {count} tasks")
        return

    # Regular mode with specific task IDs
    if args.task_ids is None:
        print("ERROR: Must provide either --task_ids or --scan_first")
        sys.exit(1)

    # Parse task IDs
    task_ids = [tid.strip() for tid in args.task_ids.split(',')]

    print("=" * 70)
    print("M2 DEBUG: Θ Feasibility Analysis")
    print("=" * 70)
    print(f"Task IDs: {task_ids}")
    print()

    # Load tasks
    tasks = load_specific_tasks(data_dir, task_ids)

    print(f"Loaded {len(tasks)} tasks")
    print("=" * 70)
    print()

    # Process each task
    for task_id in task_ids:
        if task_id not in tasks:
            print(f"TASK {task_id}: NOT FOUND")
            print()
            continue

        pairs = tasks[task_id]

        print(f"TASK {task_id}")
        print("-" * 70)

        # Step 1: Apply Π and show shapes
        trains_pi = []
        shapes_Xpi = []
        shapes_Ypi = []

        for X, Y in pairs:
            X_canon = canon_orient(X)
            Y_canon = canon_orient(Y)
            trains_pi.append((X_canon.grid, Y_canon.grid))

            h_x, w_x = len(X_canon.grid), len(X_canon.grid[0]) if X_canon.grid else 0
            h_y, w_y = len(Y_canon.grid), len(Y_canon.grid[0]) if Y_canon.grid else 0

            shapes_Xpi.append((h_x, w_x))
            shapes_Ypi.append((h_y, w_y))

        print(f"shapes_Xπ: {shapes_Xpi}")
        print(f"shapes_Yπ: {shapes_Ypi}")

        # Step 2: Get per-pair candidates
        per_pair_candidates = get_per_pair_candidates(trains_pi)

        for i, candidates in enumerate(per_pair_candidates):
            cand_kinds = [c.kind for c in candidates]
            print(f"CAND_{i}: {cand_kinds}")

        # Step 3: Get intersection (final Θ_feas)
        thetas_feas = enumerate_feasible_P(trains_pi)
        theta_kinds = [t.kind for t in thetas_feas]
        print(f"Θ_feas: {theta_kinds}")

        # Step 4: Sanity checks
        all_same_shape = all(shapes_Xpi[i] == shapes_Ypi[i] for i in range(len(pairs)))
        all_transpose_shape = all(
            (shapes_Xpi[i][0], shapes_Xpi[i][1]) == (shapes_Ypi[i][1], shapes_Ypi[i][0])
            for i in range(len(pairs))
        )

        print(f"ALL_SAME_SHAPE: {all_same_shape}", end="  ")
        if all_same_shape:
            identity_present = "identity" in theta_kinds
            print(f"CHECK_IDENTITY: {'PASS' if identity_present else 'FAIL'}")
        else:
            print()

        print(f"ALL_TRANSPOSE_SHAPE: {all_transpose_shape}", end="  ")
        if all_transpose_shape:
            transpose_present = "transpose" in theta_kinds
            print(f"CHECK_TRANSPOSE: {'PASS' if transpose_present else 'FAIL'}")
        else:
            print()

        # Step 5: Empty Θ reason
        if not thetas_feas:
            # Check if any per-pair candidate list is empty
            empty_pairs = [i for i, cands in enumerate(per_pair_candidates) if not cands]

            if empty_pairs:
                print(f"EMPTY_THETA_REASON: PAIR_ZERO_CANDIDATES (pairs: {empty_pairs})")
            elif all(per_pair_candidates):
                print(f"EMPTY_THETA_REASON: NONEMPTY_INTERSECTION_EMPTY")
            else:
                print(f"EMPTY_THETA_REASON: UNKNOWN")
        else:
            print(f"EMPTY_THETA_REASON: NONE")

        print()
        print("=" * 70)
        print()


if __name__ == '__main__':
    main()
