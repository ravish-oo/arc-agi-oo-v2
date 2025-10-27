#!/usr/bin/env python3
"""
M5 End-to-End Verifier: Real Test on Actual ARC Challenges

Tests the full pipeline:
1. Learn rulebook from training pairs
2. Apply to test inputs
3. Compare with actual test solutions

This is the REAL measure of success.
"""

import sys
import json
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient
from src.p_menu import enumerate_feasible_P, apply_theta
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.fy import learn_rules_via_wl_and_actions, glue_once, build_row_col_blocks, AuxData
from src.utils import dims, same_grid


def load_training_tasks(data_dir: Path, max_tasks: int = 20):
    """
    Load ARC training tasks with both challenges and solutions.

    Filters to tasks with:
    - len(train) ∈ {1, 2, 3} (small for testing)
    - At least one feasible theta
    - Has test cases with solutions

    Returns dict of task_id -> (train_pairs, test_pairs, test_solutions)
    """
    challenges_path = data_dir / "arc-agi_training_challenges.json"
    solutions_path = data_dir / "arc-agi_training_solutions.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    if not solutions_path.exists():
        print(f"ERROR: {solutions_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    tasks = {}
    for task_id, task in sorted(challenges.items()):
        train_pairs = task.get('train', [])
        test_pairs = task.get('test', [])

        if len(train_pairs) not in {1, 2, 3}:
            continue

        if len(test_pairs) == 0:
            continue

        if task_id not in solutions:
            continue

        # Convert to (X, Y) format
        train = [(p['input'], p['output']) for p in train_pairs]
        test_inputs = [p['input'] for p in test_pairs]
        test_outputs = solutions[task_id]  # Ground truth

        # Check if at least one feasible theta exists
        trains_pi = []
        for X, Y in train:
            Xc = canon_orient(X).grid
            Yc = canon_orient(Y).grid
            trains_pi.append((Xc, Yc))

        thetas = enumerate_feasible_P(trains_pi)
        if len(thetas) == 0:
            continue

        tasks[task_id] = {
            'train': train,
            'test_inputs': test_inputs,
            'test_solutions': test_outputs
        }

        if len(tasks) >= max_tasks:
            break

    return tasks


def apply_learned_program(test_input, theta, rulebook):
    """
    Apply learned program to a test input.

    Returns predicted output grid.
    """
    # Π: Canonize
    canon = canon_orient(test_input)
    Xc = canon.grid

    # P: Apply global transform
    Xp = apply_theta(Xc, theta)

    # Φ: WL partition
    rel = build_rel_structure(Xp)
    labels, _, _ = wl_refine(rel)

    # FY + GLUE: Apply rulebook
    h, w = dims(Xp)
    row_blocks, col_blocks = build_row_col_blocks(labels, h, w)
    aux = AuxData(row_blocks=row_blocks, col_blocks=col_blocks)
    Yp = glue_once(Xp, labels, rulebook, aux)

    # Π⁻¹: Undo canonization
    Y = canon.undo(Yp)

    return Y


def main():
    print("\n" + "=" * 60)
    print("M5 END-TO-END VERIFICATION")
    print("Learn from train, predict test, compare with solutions")
    print("=" * 60 + "\n")

    # Load tasks
    data_dir = Path(__file__).parent.parent / "data"
    max_tasks = 20
    print(f"Loading up to {max_tasks} suitable training tasks...")
    tasks = load_training_tasks(data_dir, max_tasks)
    print(f"Loaded {len(tasks)} tasks\n")

    if len(tasks) == 0:
        print("ERROR: No suitable tasks found")
        return 1

    # Process tasks
    print("=" * 60)
    print("PROCESSING TASKS")
    print("=" * 60)

    tasks_total = 0
    tasks_learned = 0  # Successfully learned from training
    tasks_correct = 0  # Correctly predicted all test cases
    tasks_partial = 0  # Some test cases correct

    exact_examples = []
    failed_learning = []
    failed_generalization = []

    for task_id, task_data in tasks.items():
        tasks_total += 1
        train = task_data['train']
        test_inputs = task_data['test_inputs']
        test_solutions = task_data['test_solutions']

        # Step 1: Learn from training data
        trains_pi = []
        for X, Y in train:
            Xc = canon_orient(X).grid
            Yc = canon_orient(Y).grid
            trains_pi.append((Xc, Yc))

        thetas = enumerate_feasible_P(trains_pi)
        result = learn_rules_via_wl_and_actions(train, thetas, escalate_policy=None)

        if not result.ok:
            print(f"✗ Task {task_id}: Failed to learn (reason: {result.witness.get('reason', 'unknown')})")
            failed_learning.append(task_id)
            continue

        tasks_learned += 1
        theta = result.theta
        rulebook = result.rulebook

        # Step 2: Apply to test cases
        test_correct = 0
        for i, (test_input, expected_output) in enumerate(zip(test_inputs, test_solutions)):
            try:
                predicted = apply_learned_program(test_input, theta, rulebook)
                if same_grid(predicted, expected_output):
                    test_correct += 1
            except Exception as e:
                # Prediction failed
                pass

        # Step 3: Check results
        if test_correct == len(test_inputs):
            tasks_correct += 1
            print(f"✓ Task {task_id}: ALL {len(test_inputs)} test cases correct!")
            print(f"  Theta: {theta.kind}, Rulebook size: {len(rulebook)}")
            if len(exact_examples) < 3:
                exact_examples.append(task_id)
        elif test_correct > 0:
            tasks_partial += 1
            print(f"◐ Task {task_id}: {test_correct}/{len(test_inputs)} test cases correct")
            failed_generalization.append((task_id, test_correct, len(test_inputs)))
        else:
            print(f"✗ Task {task_id}: Learned, but 0/{len(test_inputs)} test cases correct")
            failed_generalization.append((task_id, 0, len(test_inputs)))

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks: {tasks_total}")
    print(f"Learned successfully: {tasks_learned} ({100*tasks_learned/tasks_total:.1f}%)")
    print(f"All tests correct: {tasks_correct} ({100*tasks_correct/tasks_total:.1f}%)")
    print(f"Some tests correct: {tasks_partial}")
    print(f"Failed to learn: {len(failed_learning)}")
    print(f"Failed generalization: {len(failed_generalization)}")
    print()

    if exact_examples:
        print("Examples of FULLY SOLVED tasks:")
        for task_id in exact_examples:
            print(f"  {task_id}")
        print()

    if failed_learning:
        print("Failed to learn (first 3):")
        for task_id in failed_learning[:3]:
            print(f"  {task_id}")
        print()

    if failed_generalization:
        print("Learned but failed generalization (first 3):")
        for task_id, correct, total in failed_generalization[:3]:
            print(f"  {task_id}: {correct}/{total} test cases")
        print()

    # Exit criteria
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if tasks_correct > 0:
        print(f"✓ SUCCESS: {tasks_correct}/{tasks_total} tasks FULLY SOLVED")
        print(f"  Real solve rate: {100*tasks_correct/tasks_total:.1f}%")
        return 0
    else:
        print(f"✗ FAILURE: No tasks fully solved")
        return 1


if __name__ == '__main__':
    sys.exit(main())
