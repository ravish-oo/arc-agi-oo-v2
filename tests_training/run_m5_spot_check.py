#!/usr/bin/env python3
"""
M5 Spot Check: Test 5 specific tasks end-to-end

Pick 5 tasks that passed M5b training sanity check and see if they
actually generalize to test cases.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient, apply_transform
from src.p_menu import enumerate_feasible_P, apply_theta
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine
from src.fy import learn_rules_via_wl_and_actions, glue_once, build_row_col_blocks, AuxData
from src.utils import dims, same_grid


# Tasks that passed M5b training sanity check
TARGET_TASKS = [
    "00576224",  # scale_up
    "025d127b",  # identity
    "03560426",  # identity
    "0692e18c",  # scale_up
    "06df4c85",  # R270
]


def apply_learned_program(test_input, theta, rulebook):
    """Apply learned program to a test input."""
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
    Y = apply_transform(Yp, canon.undo_code)

    return Y


def main():
    print("\n" + "=" * 60)
    print("M5 SPOT CHECK: 5 Tasks End-to-End")
    print("=" * 60 + "\n")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    challenges_path = data_dir / "arc-agi_training_challenges.json"
    solutions_path = data_dir / "arc-agi_training_solutions.json"

    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    total_correct = 0
    total_tests = 0

    for task_id in TARGET_TASKS:
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_id}")
        print("=" * 60)

        if task_id not in challenges or task_id not in solutions:
            print(f"✗ Task data not found")
            continue

        task = challenges[task_id]
        train_pairs = [(p['input'], p['output']) for p in task['train']]
        test_inputs = [p['input'] for p in task['test']]
        test_solutions = solutions[task_id]

        print(f"Training pairs: {len(train_pairs)}")
        print(f"Test cases: {len(test_inputs)}")
        print()

        # Step 1: Learn from training
        print("Step 1: Learning from training data...")
        trains_pi = []
        for X, Y in train_pairs:
            Xc = canon_orient(X).grid
            Yc = canon_orient(Y).grid
            trains_pi.append((Xc, Yc))

        thetas = enumerate_feasible_P(trains_pi)
        result = learn_rules_via_wl_and_actions(train_pairs, thetas, escalate_policy=None)

        if not result.ok:
            print(f"✗ Failed to learn: {result.witness.get('reason', 'unknown')}")
            continue

        theta = result.theta
        rulebook = result.rulebook
        print(f"✓ Learned successfully!")
        print(f"  Theta: {theta.kind}")
        print(f"  Rulebook size: {len(rulebook)} classes")
        print()

        # Step 2: Apply to test cases
        print("Step 2: Applying to test cases...")
        test_correct = 0
        for i, (test_input, expected) in enumerate(zip(test_inputs, test_solutions)):
            try:
                predicted = apply_learned_program(test_input, theta, rulebook)

                print(f"  Test {i+1}:")
                print(f"    Input: {len(test_input)}×{len(test_input[0])}")
                print(f"    Expected: {len(expected)}×{len(expected[0])}")
                print(f"    Predicted: {len(predicted)}×{len(predicted[0])}")

                if same_grid(predicted, expected):
                    print(f"    ✓ MATCH!")
                    test_correct += 1
                    total_correct += 1
                else:
                    print(f"    ✗ MISMATCH")
                    # Show first difference
                    for r in range(min(len(expected), len(predicted))):
                        for c in range(min(len(expected[0]), len(predicted[0]) if predicted else 0)):
                            if predicted[r][c] != expected[r][c]:
                                print(f"    First diff at ({r},{c}): expected {expected[r][c]}, got {predicted[r][c]}")
                                break
                        else:
                            continue
                        break

                total_tests += 1
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                total_tests += 1

        print()
        print(f"Result: {test_correct}/{len(test_inputs)} test cases correct")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Total test cases: {total_tests}")
    print(f"Correct predictions: {total_correct}")
    if total_tests > 0:
        print(f"Accuracy: {100*total_correct/total_tests:.1f}%")
    print()

    if total_correct == total_tests and total_tests > 0:
        print("✓ ALL TEST CASES PASSED!")
        return 0
    elif total_correct > 0:
        print(f"◐ PARTIAL SUCCESS: {total_correct}/{total_tests} passed")
        return 0
    else:
        print("✗ NO TEST CASES PASSED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
