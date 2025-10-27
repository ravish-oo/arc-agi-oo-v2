#!/usr/bin/env python3
"""
M6 Verifier: Full pipeline with training equality gate + MDL + test prediction.

Tests solve_task() on ARC training tasks:
  1. Load tasks with 1-3 train pairs
  2. Call solve_task(trains, tests)
  3. If ok=True: verify bit-exact training equality, print diagnostics
  4. If ok=False: assert witness present, print reason
  5. Summary: stats, θ distribution, determinism check
"""

import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solve import solve_task
from src.pi_orient import canon_orient
from src.p_menu import apply_theta
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.glue import glue_once
from src.fy import mk_rowcol_blocks, AuxData
from src.utils import dims, same_grid


def load_tasks(
    kaggle_path: Path,
    max_tasks: int,
    require_theta: bool,
    max_grid_size: int = 900
):
    """
    Load ARC training tasks.

    Filters to tasks with 1-3 train pairs.
    If require_theta=True, only tasks with non-empty Θ.
    Skip tasks with grids larger than max_grid_size pixels.
    """
    from src.p_menu import enumerate_feasible_P

    challenges_path = kaggle_path / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    tasks = {}
    for task_id in sorted(data.keys()):
        task = data[task_id]
        train_pairs = task.get('train', [])
        if len(train_pairs) not in {1, 2, 3}:
            continue

        # Convert to (X, Y) format
        trains = [(p['input'], p['output']) for p in train_pairs]

        # Skip large grids
        max_size = max(
            len(grid) * len(grid[0]) if len(grid) > 0 else 0
            for pair in train_pairs
            for grid in [pair['input'], pair['output']]
        )
        if max_size > max_grid_size:
            continue

        # Check for feasible theta if required
        if require_theta:
            trains_pi = []
            for X, Y in trains:
                Xc = canon_orient(X).grid
                Yc = canon_orient(Y).grid
                trains_pi.append((Xc, Yc))

            thetas = enumerate_feasible_P(trains_pi)
            if len(thetas) == 0:
                continue

        # Get test inputs
        test_inputs = [p['input'] for p in task.get('test', [])]

        tasks[task_id] = {
            'trains': trains,
            'tests': test_inputs
        }

        if len(tasks) >= max_tasks:
            break

    return tasks


def verify_training_equality(
    trains,
    theta,
    rulebook,
    escalate_policy,
    use_task_color_canon,
    lut_density_tau
):
    """
    Re-verify bit-exact training equality by running GLUE on each training pair.

    Returns True if all pairs match exactly, False otherwise.
    """
    for X, Y in trains:
        # Π
        Xc = canon_orient(X).grid
        Yc = canon_orient(Y).grid

        # P
        Xp = apply_theta(Xc, theta)

        # Φ
        rel = build_rel_structure(Xp)
        labels, _, _ = wl_refine(rel, max_iters=20, escalate=escalate_policy)

        # Aux
        h, w = dims(Xp)
        row_blocks, col_blocks = mk_rowcol_blocks(labels, h, w)
        aux = AuxData(row_blocks=row_blocks, col_blocks=col_blocks)

        # GLUE
        Yp_pred = glue_once(Xp, labels, rulebook, aux)

        # Check equality
        if not same_grid(Yp_pred, Yc):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description='M6 Verifier with Final Training Equality')
    parser.add_argument('--kaggle_path', type=str, default='data',
                       help='Path to ARC data directory')
    parser.add_argument('--max_tasks', type=int, default=30,
                       help='Maximum number of tasks to test')
    parser.add_argument('--max_grid_size', type=int, default=900,
                       help='Skip grids larger than this (pixels)')
    parser.add_argument('--require_theta', action='store_true',
                       help='Only select tasks with non-empty Θ')
    parser.add_argument('--use_task_color_canon', action='store_true',
                       help='Use task-level color canonicalization')
    parser.add_argument('--lut_density_tau', type=float, default=0.8,
                       help='LUT density threshold for Gate D')
    parser.add_argument('--escalate_policy', type=str, default=None,
                       help='Escalation policy (None, E8, 2WL)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("M6 VERIFIER: Full Pipeline with Training Equality")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  kaggle_path: {args.kaggle_path}")
    print(f"  max_tasks: {args.max_tasks}")
    print(f"  max_grid_size: {args.max_grid_size}")
    print(f"  require_theta: {args.require_theta}")
    print(f"  lut_density_tau: {args.lut_density_tau}")
    print(f"  escalate_policy: {args.escalate_policy}")
    print()

    # Load tasks
    kaggle_path = Path(args.kaggle_path)
    print(f"Loading tasks from {kaggle_path}...")
    tasks = load_tasks(
        kaggle_path,
        args.max_tasks,
        args.require_theta,
        args.max_grid_size
    )
    print(f"Loaded {len(tasks)} tasks (skipped large grids >{args.max_grid_size}px)\n")

    if len(tasks) == 0:
        print("ERROR: No suitable tasks found")
        return 1

    # Process tasks
    print("=" * 70)
    print("PROCESSING TASKS")
    print("=" * 70)

    tasks_total = 0
    tasks_exact = 0
    tasks_unsat = 0

    theta_distribution = Counter()
    action_distribution = Counter()
    num_classes_list = []
    lut_keys_list = []

    exact_examples = []
    unsat_examples = []

    for task_id, task_data in sorted(tasks.items()):
        tasks_total += 1
        trains = task_data['trains']
        tests = task_data['tests']

        # Progress indicator
        print(f"Processing {task_id} ({tasks_total}/{len(tasks)})...")

        # Call solve_task
        result = solve_task(
            trains,
            tests,
            escalate_policy=args.escalate_policy,
            use_task_color_canon=args.use_task_color_canon,
            lut_density_tau=args.lut_density_tau
        )

        # Determinism check: run twice and compare
        result2 = solve_task(
            trains,
            tests,
            escalate_policy=args.escalate_policy,
            use_task_color_canon=args.use_task_color_canon,
            lut_density_tau=args.lut_density_tau
        )

        # Verify results match
        if result.ok != result2.ok:
            print(f"✗ {task_id}: DETERMINISM FAILURE")
            print(f"    ERROR: First run ok={result.ok}, second run ok={result2.ok}")
            sys.exit(1)

        if result.ok:
            # Compare theta, rulebook, and predictions
            if result.theta != result2.theta:
                print(f"✗ {task_id}: DETERMINISM FAILURE")
                print(f"    ERROR: Theta mismatch between runs")
                sys.exit(1)

            if result.rulebook != result2.rulebook:
                print(f"✗ {task_id}: DETERMINISM FAILURE")
                print(f"    ERROR: Rulebook mismatch between runs")
                sys.exit(1)

            if not same_grid(result.preds[0] if result.preds else [],
                           result2.preds[0] if result2.preds else []):
                print(f"✗ {task_id}: DETERMINISM FAILURE")
                print(f"    ERROR: Predictions mismatch between runs")
                sys.exit(1)
        else:
            # Compare witness
            if result.witness != result2.witness:
                print(f"✗ {task_id}: DETERMINISM FAILURE")
                print(f"    ERROR: Witness mismatch between runs")
                sys.exit(1)

        if result.ok:
            tasks_exact += 1

            # Verify training equality
            equality_ok = verify_training_equality(
                trains,
                result.theta,
                result.rulebook,
                args.escalate_policy,
                args.use_task_color_canon,
                args.lut_density_tau
            )

            if not equality_ok:
                print(f"✗ {task_id}: EQUALITY VERIFICATION FAILED")
                print(f"    ERROR: GLUE does not reproduce training outputs!")
                sys.exit(1)

            # Statistics
            theta_kind = result.theta.kind if result.theta else "identity"
            theta_distribution[theta_kind] += 1
            num_classes = len(result.rulebook)
            num_classes_list.append(num_classes)

            # Count LUT keys
            lut_keys = 0
            for rule in result.rulebook.values():
                if rule.action in ["lut_r2", "lut_r3"]:
                    lut = rule.params.get("lut", {})
                    lut_keys += len(lut)
            lut_keys_list.append(lut_keys)

            # Action distribution (top 3 per task)
            task_actions = Counter()
            for rule in result.rulebook.values():
                task_actions[rule.action] += 1
                action_distribution[rule.action] += 1

            print(f"✓ {task_id}: EXACT")
            print(f"    theta={theta_kind}, classes={num_classes}, lut_keys={lut_keys}")
            top3_actions = task_actions.most_common(3)
            print(f"    Top actions: {dict(top3_actions)}")

            if len(exact_examples) < 3:
                exact_examples.append((task_id, theta_kind, num_classes))

        else:
            tasks_unsat += 1
            witness = result.witness
            reason = witness.get("reason", "unknown") if witness else "no_witness"
            class_id = witness.get("class_id", "?") if witness else "?"

            print(f"✗ {task_id}: UNSAT")
            print(f"    reason={reason}, class={class_id}")

            if len(unsat_examples) < 3:
                unsat_examples.append((task_id, reason))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tasks: {tasks_total}")
    print(f"Exact (passed training equality): {tasks_exact} ({100*tasks_exact/tasks_total if tasks_total > 0 else 0:.1f}%)")
    print(f"UNSAT: {tasks_unsat}")
    print()

    # Theta distribution
    if theta_distribution:
        print("Theta Distribution:")
        for theta_kind, count in sorted(theta_distribution.items(), key=lambda x: -x[1]):
            print(f"  {theta_kind:20s}: {count:3d}")
        print()

    # Action distribution
    if action_distribution:
        print("Action Distribution (across all exact tasks):")
        for action, count in sorted(action_distribution.items(), key=lambda x: -x[1])[:10]:
            print(f"  {action:30s}: {count:4d}")
        print()

    # Average stats
    if num_classes_list:
        avg_classes = sum(num_classes_list) / len(num_classes_list)
        avg_lut_keys = sum(lut_keys_list) / len(lut_keys_list)
        print(f"Average classes per exact task: {avg_classes:.1f}")
        print(f"Average LUT keys per exact task: {avg_lut_keys:.1f}")
        print()

    # Examples
    if exact_examples:
        print("Exact Examples:")
        for task_id, theta_kind, num_classes in exact_examples:
            print(f"  {task_id}: theta={theta_kind}, classes={num_classes}")
        print()

    if unsat_examples:
        print("UNSAT Examples:")
        for task_id, reason in unsat_examples:
            print(f"  {task_id}: {reason}")
        print()

    # Exit criteria
    print("=" * 70)
    print("EXIT CRITERIA")
    print("=" * 70)
    print("✓ Verifier ran to completion")
    print("✓ All EXACT tasks passed training equality verification")
    print("✓ All UNSAT tasks have witness")
    print()

    if tasks_exact > 0:
        print(f"✓ {tasks_exact} tasks solved with bit-exact training equality")
    else:
        print("⚠ No tasks solved")

    print()
    print("=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
