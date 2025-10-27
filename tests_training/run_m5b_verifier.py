#!/usr/bin/env python3
"""
M5b Verifier: Test Learning Driver

Tests full end-to-end learning on real ARC training tasks:
1. Select tasks with small train sets and feasible thetas
2. Learn rulebooks from training pairs
3. Verify exact matches for ok=True
4. Check witness structure for ok=False
5. Report statistics
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient
from src.p_menu import enumerate_feasible_P
from src.fy import learn_rules_via_wl_and_actions, glue_once, class_masks, build_row_col_blocks, AuxData
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.p_menu import apply_theta
from src.utils import dims, same_grid


def load_training_tasks(data_dir: Path, max_tasks: int = 40):
    """
    Load ARC training tasks.

    Filters to tasks with:
    - len(train) ∈ {1, 2, 3} (small for testing)
    - At least one feasible theta

    Returns dict of task_id -> list of (X, Y) pairs
    """
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    tasks = {}
    for task_id, task in sorted(data.items()):
        train_pairs = task.get('train', [])
        if len(train_pairs) not in {1, 2, 3}:
            continue

        pairs = [(p['input'], p['output']) for p in train_pairs]

        # Check if at least one feasible theta exists (quick filter)
        # Canonize pairs
        trains_pi = []
        for X, Y in pairs:
            Xc = canon_orient(X).grid
            Yc = canon_orient(Y).grid
            trains_pi.append((Xc, Yc))

        thetas = enumerate_feasible_P(trains_pi)
        if len(thetas) == 0:
            continue  # Skip tasks with no feasible theta

        tasks[task_id] = pairs

        if len(tasks) >= max_tasks:
            break

    return tasks


def main():
    print("\n" + "=" * 60)
    print("M5b VERIFICATION: Learning Driver")
    print("=" * 60 + "\n")

    # Open output file for incremental progress
    output_file = Path(__file__).parent.parent / "m5b_progress.log"
    with open(output_file, 'w') as log:
        log.write("=" * 60 + "\n")
        log.write("M5b VERIFICATION: Learning Driver\n")
        log.write("=" * 60 + "\n\n")
        log.flush()

    # Load tasks
    data_dir = Path(__file__).parent.parent / "data"
    max_tasks = 20  # Test with 20 tasks
    msg = f"Loading up to {max_tasks} suitable training tasks..."
    print(msg)
    with open(output_file, 'a') as log:
        log.write(msg + "\n")
        log.flush()

    tasks = load_training_tasks(data_dir, max_tasks)
    msg = f"Loaded {len(tasks)} tasks\n"
    print(msg)
    with open(output_file, 'a') as log:
        log.write(msg + "\n")
        log.flush()

    if len(tasks) == 0:
        print("ERROR: No suitable tasks found")
        return 1

    # Process tasks
    print("=" * 60)
    print("PROCESSING TASKS")
    print("=" * 60)
    with open(output_file, 'a') as log:
        log.write("=" * 60 + "\n")
        log.write("PROCESSING TASKS\n")
        log.write("=" * 60 + "\n")
        log.flush()

    tasks_total = 0
    tasks_exact = 0
    tasks_unsat = 0
    action_counts = Counter()
    exact_examples = []
    unsat_examples = []

    for task_id, pairs in tasks.items():
        tasks_total += 1

        # Canonize pairs
        trains_pi = []
        for X, Y in pairs:
            Xc = canon_orient(X).grid
            Yc = canon_orient(Y).grid
            trains_pi.append((Xc, Yc))

        # Get feasible thetas
        thetas = enumerate_feasible_P(trains_pi)

        # Learn
        result = learn_rules_via_wl_and_actions(pairs, thetas, escalate_policy=None)

        if result.ok:
            tasks_exact += 1
            msg = f"✓ Task {task_id}: SUCCESS"
            print(msg)
            with open(output_file, 'a') as log:
                log.write(msg + "\n")
                log.write(f"  Theta: {result.theta.kind if result.theta else '?'}\n")
                log.write(f"  Rulebook size: {len(result.rulebook)}\n")
                for class_id, rule in sorted(result.rulebook.items())[:5]:
                    log.write(f"    Class {class_id}: {rule.action}\n")
                log.flush()

            print(f"  Theta: {result.theta.kind if result.theta else '?'}")
            print(f"  Rulebook size: {len(result.rulebook)}")
            for class_id, rule in sorted(result.rulebook.items())[:5]:
                print(f"    Class {class_id}: {rule.action}")

            if len(exact_examples) < 3:
                exact_examples.append((task_id, result.theta.kind if result.theta else "?", len(result.rulebook)))

            # Count actions used
            for rule in result.rulebook.values():
                action_counts[rule.action] += 1

            # Verify glue produces exact match
            theta = result.theta
            rulebook = result.rulebook

            for i, (X, Y) in enumerate(pairs):
                Xc = canon_orient(X).grid
                Yc = canon_orient(Y).grid
                Xp = apply_theta(Xc, theta)
                Yp = Yc

                # Build Φ labels
                rel = build_rel_structure(Xp)
                labels, _, _ = wl_refine(rel)

                # Build aux
                h, w = dims(Xp)
                row_blocks, col_blocks = build_row_col_blocks(labels, h, w)
                aux = AuxData(row_blocks=row_blocks, col_blocks=col_blocks)

                # Glue
                composed = glue_once(Xp, labels, rulebook, aux)

                if not same_grid(composed, Yp):
                    msg = f"✗ Task {task_id}: glue verification failed for pair {i}"
                    print(msg)
                    with open(output_file, 'a') as log:
                        log.write(msg + "\n")
                        log.flush()
                    tasks_exact -= 1
                    tasks_unsat += 1
                    break
        else:
            tasks_unsat += 1
            msg = f"✗ Task {task_id}: UNSAT"
            print(msg)
            with open(output_file, 'a') as log:
                log.write(msg + "\n")
                if result.witness:
                    reason = result.witness.get("reason", "unknown")
                    log.write(f"  Reason: {reason}\n")
                    if "class_id" in result.witness:
                        log.write(f"  Failed class: {result.witness['class_id']}\n")
                    if "mismatch_at" in result.witness:
                        log.write(f"  Mismatch at {result.witness['mismatch_at']}: expected {result.witness['expected']}, got {result.witness['got']}, class={result.witness['class']}\n")
                log.flush()

            if result.witness:
                reason = result.witness.get("reason", "unknown")
                print(f"  Reason: {reason}")
                if "class_id" in result.witness:
                    print(f"  Failed class: {result.witness['class_id']}")
                if "mismatch_at" in result.witness:
                    print(f"  Mismatch at {result.witness['mismatch_at']}: expected {result.witness['expected']}, got {result.witness['got']}, class={result.witness['class']}")
                if len(unsat_examples) < 3:
                    unsat_examples.append((task_id, reason))

    # Summary
    print(f"\nProcessed {tasks_total} tasks")
    print(f"Exact: {tasks_exact}")
    print(f"UNSAT: {tasks_unsat}")
    print()

    # Action distribution
    if action_counts:
        print("=" * 60)
        print("ACTION DISTRIBUTION")
        print("=" * 60)
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {action:25s}: {count:3d}")
        print()

    # Examples
    print("=" * 60)
    print("EXAMPLES")
    print("=" * 60)

    if exact_examples:
        print("\nExact (solved):")
        for task_id, theta_kind, num_classes in exact_examples:
            print(f"  {task_id}: theta={theta_kind}, classes={num_classes}")

    if unsat_examples:
        print("\nUNSAT:")
        for task_id, reason in unsat_examples:
            print(f"  {task_id}: {reason}")
    print()

    # Check exit criteria
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if tasks_total == 0:
        print("✗ No tasks processed")
        return 1

    solve_rate = (tasks_exact / tasks_total) * 100 if tasks_total > 0 else 0
    print(f"Solve rate: {solve_rate:.1f}% ({tasks_exact}/{tasks_total})")
    print()

    if tasks_exact > 0:
        print("✓ AT LEAST SOME TASKS SOLVED")
        return 0
    else:
        print("✗ NO TASKS SOLVED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
