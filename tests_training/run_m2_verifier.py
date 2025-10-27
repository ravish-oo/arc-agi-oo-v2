#!/usr/bin/env python3
"""
M2 Verifier: Check P (global map feasibility chooser).

Checks on training data:
1. All tasks have ≥1 feasible theta
2. All returned thetas satisfy shape compatibility on all pairs
3. p_menu.py does not contain content equality checks to Y
4. Report statistics and samples
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient
from src.p_menu import enumerate_feasible_P, output_shape_for_theta, Theta


def load_training_tasks(data_dir: Path):
    """
    Load ARC training challenges.
    Returns dict of task_id -> list of (X, Y) pairs.
    """
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    tasks = {}
    for task_id, task in data.items():
        pairs = []
        for pair in task.get('train', []):
            X = pair['input']
            Y = pair['output']
            pairs.append((X, Y))
        tasks[task_id] = pairs

    return tasks


def negative_grep_check(p_menu_path: Path):
    """
    Check that p_menu.py does not contain content equality checks to Y.

    Allowed: len(Y), len(Y[0]) for shape access
    Forbidden: Y[i][j] for pixel value access, comparisons like X == Y
    """
    with open(p_menu_path, 'r') as f:
        content = f.read()

    violations = []

    # Check each line for violations
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue

        # Look for pixel value access patterns Y[i][j] (double indexing)
        # But allow len(Y) and len(Y[0]) for shape
        if 'Y[' in line:
            # Check if it's shape access (len(Y[0])) or pixel access (Y[i][j])
            if 'len(Y[0])' in line or 'len(Y)' in line:
                # Shape access is OK
                continue

            # Look for double indexing pattern: Y[...][...]
            if re.search(r'\bY\[[^\]]+\]\[[^\]]+\]', line):
                violations.append(f"Line {line_num}: Found Y[...][...] pixel access pattern")

        # Look for content equality patterns
        # X == Y or Y == X (but not part of != or other operators)
        if re.search(r'(\bX\s*==\s*Y\b|\bY\s*==\s*X\b)', line):
            violations.append(f"Line {line_num}: Found X == Y content equality check")
        if re.search(r'(\bX\s*!=\s*Y\b|\bY\s*!=\s*X\b)', line):
            violations.append(f"Line {line_num}: Found X != Y content inequality check")

    return violations


def main():
    print("\n" + "=" * 60)
    print("M2 VERIFICATION: P (Global Map Feasibility)")
    print("=" * 60 + "\n")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    print(f"Loading tasks from {data_dir}")
    tasks = load_training_tasks(data_dir)
    print(f"Loaded {len(tasks)} training tasks\n")

    # Negative grep check
    print("=" * 60)
    print("NEGATIVE GREP CHECK (no content equality to Y)")
    print("=" * 60)
    p_menu_path = Path(__file__).parent.parent / "src" / "p_menu.py"
    violations = negative_grep_check(p_menu_path)

    if violations:
        print("VIOLATIONS FOUND:")
        for v in violations:
            print(f"  ✗ {v}")
        print("\nFailing due to negative grep violations.\n")
        return 1
    else:
        print("✓ No violations found (p_menu.py is content-blind)\n")

    # Process tasks
    print("=" * 60)
    print("PROCESSING TASKS")
    print("=" * 60)

    tasks_processed = 0
    tasks_empty_theta = []
    theta_counts = []
    theta_kinds = []
    shape_check_failures = []

    for task_id, pairs in sorted(tasks.items()):
        tasks_processed += 1

        # Canonize all pairs
        trains_pi = []
        for X, Y in pairs:
            X_canon = canon_orient(X)
            Y_canon = canon_orient(Y)
            trains_pi.append((X_canon.grid, Y_canon.grid))

        # Enumerate feasible P
        try:
            thetas = enumerate_feasible_P(trains_pi)
        except Exception as e:
            print(f"ERROR in task {task_id}: {e}")
            tasks_empty_theta.append((task_id, 0))
            continue

        theta_counts.append((task_id, len(thetas)))

        if len(thetas) == 0:
            tasks_empty_theta.append((task_id, 0))
            continue

        # Collect kinds
        for theta in thetas:
            theta_kinds.append(theta.kind)

        # Verify shape compatibility for all returned thetas
        for theta in thetas:
            for i, (X, Y) in enumerate(trains_pi):
                h_X, w_X = len(X), len(X[0]) if X else 0
                h_Y, w_Y = len(Y), len(Y[0]) if Y else 0

                out_shape = output_shape_for_theta((h_X, w_X), theta)

                if out_shape is None:
                    shape_check_failures.append({
                        'task_id': task_id,
                        'pair_idx': i,
                        'theta': theta,
                        'reason': 'output_shape_for_theta returned None'
                    })
                elif out_shape != (h_Y, w_Y):
                    shape_check_failures.append({
                        'task_id': task_id,
                        'pair_idx': i,
                        'theta': theta,
                        'expected_shape': (h_Y, w_Y),
                        'got_shape': out_shape
                    })

    print(f"Tasks processed: {tasks_processed}")
    print(f"Tasks with empty Θ: {len(tasks_empty_theta)}")

    if theta_counts:
        avg_theta = sum(count for _, count in theta_counts) / len(theta_counts)
        print(f"Average |Θ_feas|: {avg_theta:.2f}")
    else:
        print("Average |Θ_feas|: N/A (no tasks processed)")

    print()

    # Histogram of kinds
    if theta_kinds:
        print("=" * 60)
        print("HISTOGRAM OF THETA KINDS")
        print("=" * 60)
        kind_counts = Counter(theta_kinds)
        for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
            print(f"  {kind:15s}: {count:5d}")
        print()

    # Sample tasks with smallest and largest |Θ|
    print("=" * 60)
    print("SAMPLE TASKS")
    print("=" * 60)

    if theta_counts:
        # Sort by count
        sorted_counts = sorted(theta_counts, key=lambda x: x[1])

        print("\nSmallest |Θ_feas| (first 3):")
        for task_id, count in sorted_counts[:3]:
            print(f"  {task_id}: {count} thetas")

        print("\nLargest |Θ_feas| (last 3):")
        for task_id, count in sorted_counts[-3:]:
            print(f"  {task_id}: {count} thetas")
        print()

    # Check for failures
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    all_passed = True

    if tasks_empty_theta:
        print(f"✗ {len(tasks_empty_theta)} tasks with empty Θ_feas:")
        for task_id, _ in tasks_empty_theta[:5]:
            print(f"    {task_id}")
        if len(tasks_empty_theta) > 5:
            print(f"    ... and {len(tasks_empty_theta) - 5} more")
        all_passed = False
    else:
        print(f"✓ All tasks have ≥1 feasible theta")

    if shape_check_failures:
        print(f"✗ {len(shape_check_failures)} shape compatibility failures:")
        for failure in shape_check_failures[:5]:
            print(f"    Task {failure['task_id']}, pair {failure['pair_idx']}")
            print(f"      Theta: {failure['theta']}")
            if 'reason' in failure:
                print(f"      Reason: {failure['reason']}")
            else:
                print(f"      Expected: {failure['expected_shape']}, Got: {failure['got_shape']}")
        if len(shape_check_failures) > 5:
            print(f"    ... and {len(shape_check_failures) - 5} more")
        all_passed = False
    else:
        print(f"✓ All returned thetas satisfy shape compatibility")

    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
