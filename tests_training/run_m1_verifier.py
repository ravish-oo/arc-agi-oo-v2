#!/usr/bin/env python3
"""
M1 Verifier: Check Π (orientation canon) properties.

Checks on training data:
1. Idempotence: canon(canon(Z).grid).grid == canon(Z).grid
2. Round-trip: apply(canon(Z).grid, canon(Z).undo_code) == Z
3. Determinism: running canon twice yields identical results
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient, apply_transform
from src.utils import same_grid, grid_to_string


# Fallback toy grids if no ARC data available
TOY_GRIDS = [
    [[0, 1, 2], [3, 4, 5]],
    [[1, 2], [3, 4]],
    [[5]],
    [[0, 0], [0, 1], [1, 1]],
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[2, 2, 2], [2, 1, 2], [2, 2, 2]],
    [[0, 1], [2, 3], [4, 5], [6, 7]],
    [[9, 8, 7, 6], [5, 4, 3, 2], [1, 0, 1, 2]],
]


def load_grids():
    """
    Load grids from ARC training data if available, else use toy grids.
    Returns list of grids.
    """
    arc_path = Path(__file__).parent.parent / "data" / "arc-agi_training_challenges.json"

    if arc_path.exists():
        print(f"Loading grids from {arc_path}")
        with open(arc_path, 'r') as f:
            data = json.load(f)

        grids = []
        for task_id, task in data.items():
            # Collect all training input and output grids
            for pair in task.get('train', []):
                if 'input' in pair:
                    grids.append(pair['input'])
                if 'output' in pair:
                    grids.append(pair['output'])
            # Also collect test inputs
            for pair in task.get('test', []):
                if 'input' in pair:
                    grids.append(pair['input'])

        print(f"Loaded {len(grids)} grids from ARC training data\n")
        return grids
    else:
        print("ARC data not found, using toy grids\n")
        return TOY_GRIDS


def check_idempotence(grids):
    """
    Check: canon(canon(Z).grid).grid == canon(Z).grid
    """
    print("=" * 60)
    print("Checking IDEMPOTENCE")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    for i, Z in enumerate(grids):
        a = canon_orient(Z)
        b = canon_orient(a.grid)

        if same_grid(a.grid, b.grid):
            passed += 1
        else:
            failed += 1
            failures.append({
                'grid_idx': i,
                'first_canon_str': grid_to_string(a.grid)[:80],
                'second_canon_str': grid_to_string(b.grid)[:80],
            })

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    if failures:
        print("FAILURES:")
        for f in failures[:5]:  # Show first 5 failures
            print(f"  Grid {f['grid_idx']}:")
            print(f"    First canon:  {f['first_canon_str']}")
            print(f"    Second canon: {f['second_canon_str']}")
        print()

    return passed, failed


def check_roundtrip(grids):
    """
    Check: apply(canon(Z).grid, canon(Z).undo_code) == Z
    """
    print("=" * 60)
    print("Checking ROUND-TRIP")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    for i, Z in enumerate(grids):
        a = canon_orient(Z)
        Z2 = apply_transform(a.grid, a.undo_code)

        if same_grid(Z, Z2):
            passed += 1
        else:
            failed += 1
            h1, w1 = len(Z), len(Z[0]) if Z else 0
            h2, w2 = len(Z2), len(Z2[0]) if Z2 else 0
            failures.append({
                'grid_idx': i,
                'undo_code': a.undo_code,
                'original_dims': (h1, w1),
                'roundtrip_dims': (h2, w2),
                'original_str': grid_to_string(Z)[:80],
                'roundtrip_str': grid_to_string(Z2)[:80],
            })

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    if failures:
        print("FAILURES:")
        for f in failures[:5]:  # Show first 5 failures
            print(f"  Grid {f['grid_idx']}:")
            print(f"    Undo code: {f['undo_code']}")
            print(f"    Original dims:  {f['original_dims']}")
            print(f"    Roundtrip dims: {f['roundtrip_dims']}")
            print(f"    Original:  {f['original_str']}")
            print(f"    Roundtrip: {f['roundtrip_str']}")
        print()

    return passed, failed


def check_determinism(grids):
    """
    Check: running canon twice on the same grid yields identical results
    """
    print("=" * 60)
    print("Checking DETERMINISM")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    for i, Z in enumerate(grids):
        a1 = canon_orient(Z)
        a2 = canon_orient(Z)

        if same_grid(a1.grid, a2.grid) and a1.undo_code == a2.undo_code:
            passed += 1
        else:
            failed += 1
            failures.append({
                'grid_idx': i,
                'first_undo': a1.undo_code,
                'second_undo': a2.undo_code,
                'grids_match': same_grid(a1.grid, a2.grid),
            })

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    if failures:
        print("FAILURES:")
        for f in failures[:5]:
            print(f"  Grid {f['grid_idx']}:")
            print(f"    First undo:  {f['first_undo']}")
            print(f"    Second undo: {f['second_undo']}")
            print(f"    Grids match: {f['grids_match']}")
        print()

    return passed, failed


def main():
    print("\n" + "=" * 60)
    print("M1 VERIFICATION: Π (Orientation Canon)")
    print("=" * 60 + "\n")

    grids = load_grids()

    # Run all checks
    idem_pass, idem_fail = check_idempotence(grids)
    round_pass, round_fail = check_roundtrip(grids)
    det_pass, det_fail = check_determinism(grids)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total grids checked: {len(grids)}")
    print()
    print(f"Idempotence:  {idem_pass} pass, {idem_fail} fail")
    print(f"Round-trip:   {round_pass} pass, {round_fail} fail")
    print(f"Determinism:  {det_pass} pass, {det_fail} fail")
    print()

    total_checks = idem_pass + round_pass + det_pass
    total_fails = idem_fail + round_fail + det_fail

    if total_fails == 0:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print(f"✗ {total_fails} CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
