#!/usr/bin/env python3
"""
M5a Verifier: Test Action Execution Engine

Tests all M5a components with manually crafted examples:
1. class_masks - label to mask conversion
2. build_row_col_blocks - equivalence grouping
3. build_lut_from_evidence - LUT building with collision detection
4. apply_action - action dispatcher
5. glue_once - composition of multiple edits
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fy import (
    class_masks, build_row_col_blocks, build_lut_from_evidence,
    apply_action, glue_once, Rule, Rulebook, AuxData
)
from src.utils import dims


def test_class_masks():
    """Test class_masks helper."""
    print("=" * 60)
    print("TESTING class_masks")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Simple 3x3 grid with 3 classes
    labels = [0, 0, 1, 0, 2, 1, 2, 2, 1]  # Row-major
    h, w = 3, 3

    masks = class_masks(labels, h, w)

    # Check class 0 mask
    expected_0 = [[True, True, False], [True, False, False], [False, False, False]]
    if masks[0] == expected_0:
        passed += 1
    else:
        print(f"✗ class_masks: class 0 mask incorrect")
        failed += 1

    # Check class 1 mask
    expected_1 = [[False, False, True], [False, False, True], [False, False, True]]
    if masks[1] == expected_1:
        passed += 1
    else:
        print(f"✗ class_masks: class 1 mask incorrect")
        failed += 1

    # Check class 2 mask
    expected_2 = [[False, False, False], [False, True, False], [True, True, False]]
    if masks[2] == expected_2:
        passed += 1
    else:
        print(f"✗ class_masks: class 2 mask incorrect")
        failed += 1

    print(f"class_masks tests: {passed} passed, {failed} failed\n")
    return passed, failed


def test_build_row_col_blocks():
    """Test build_row_col_blocks helper."""
    print("=" * 60)
    print("TESTING build_row_col_blocks")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test: 3x3 grid with row equivalences
    # Rows 0 and 2 have same signature, row 1 different
    labels = [
        0, 1, 0,  # Row 0
        2, 2, 2,  # Row 1 (different)
        0, 1, 0   # Row 2 (same as row 0)
    ]
    h, w = 3, 3

    row_blocks, col_blocks = build_row_col_blocks(labels, h, w)

    # Rows 0 and 2 should be in same block
    found_row_block = False
    for block in row_blocks:
        if set(block) == {0, 2}:
            found_row_block = True

    if found_row_block:
        passed += 1
    else:
        print(f"✗ build_row_col_blocks: rows 0,2 not in same block")
        failed += 1

    # Columns 0 and 2 should be in same block (both have [0, 2, 0])
    found_col_block = False
    for block in col_blocks:
        if set(block) == {0, 2}:
            found_col_block = True

    if found_col_block:
        passed += 1
    else:
        print(f"✗ build_row_col_blocks: cols 0,2 not in same block")
        failed += 1

    print(f"build_row_col_blocks tests: {passed} passed, {failed} failed\n")
    return passed, failed


def test_build_lut_from_evidence():
    """Test build_lut_from_evidence helper."""
    print("=" * 60)
    print("TESTING build_lut_from_evidence")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Simple LUT without collision
    Xp = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]
    Yp = [[5, 5, 5], [5, 9, 5], [5, 5, 5]]
    labels = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # Class 1 at center

    evidence = [(0, 1, 1)]  # Pair 0, center pixel
    lut = build_lut_from_evidence(evidence, [Xp], [Yp], [labels], class_id=1, r=2)

    if lut is not None:
        passed += 1
    else:
        print("✗ build_lut_from_evidence: returned None on valid input")
        failed += 1

    # Test 2: LUT with collision (different targets for same key)
    Xp1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    Yp1 = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    Xp2 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]  # Same pattern
    Yp2 = [[0, 0, 0], [0, 7, 0], [0, 0, 0]]  # Different target!

    labels1 = [0, 0, 0, 0, 1, 0, 0, 0, 0]
    labels2 = [0, 0, 0, 0, 1, 0, 0, 0, 0]

    evidence_collision = [(0, 1, 1), (1, 1, 1)]
    lut_collision = build_lut_from_evidence(
        evidence_collision, [Xp1, Xp2], [Yp1, Yp2], [labels1, labels2],
        class_id=1, r=2
    )

    if lut_collision is None:
        passed += 1
    else:
        print("✗ build_lut_from_evidence: should detect collision")
        failed += 1

    print(f"build_lut_from_evidence tests: {passed} passed, {failed} failed\n")
    return passed, failed


def test_apply_action():
    """Test apply_action dispatcher."""
    print("=" * 60)
    print("TESTING apply_action")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mask = [[True, True, True], [True, True, True], [True, True, True]]

    # Test mirror_h
    rule_mirror = Rule(action="mirror_h", params={})
    aux = AuxData(row_blocks=[], col_blocks=[])
    result = apply_action(grid, mask, rule_mirror, aux)

    # After horizontal mirror, first row should be [3, 2, 1]
    if result[0][0] == 3 and result[0][1] == 2 and result[0][2] == 1:
        passed += 1
    else:
        print(f"✗ apply_action: mirror_h failed")
        failed += 1

    # Test shift
    rule_shift = Rule(action="shift", params={"dr": 1, "dc": 0})
    result_shift = apply_action(grid, mask, rule_shift, aux)

    # After shift down by 1, top row should be all 0
    if result_shift[0] == [0, 0, 0]:
        passed += 1
    else:
        print(f"✗ apply_action: shift failed")
        failed += 1

    # Test set_color
    rule_color = Rule(action="set_color", params={"color": 7})
    result_color = apply_action(grid, mask, rule_color, aux)

    # All masked cells should be 7
    if all(result_color[r][c] == 7 for r in range(3) for c in range(3)):
        passed += 1
    else:
        print(f"✗ apply_action: set_color failed")
        failed += 1

    print(f"apply_action tests: {passed} passed, {failed} failed\n")
    return passed, failed


def test_glue_once():
    """Test glue_once composition."""
    print("=" * 60)
    print("TESTING glue_once")
    print("=" * 60)

    passed = 0
    failed = 0

    # Create a grid with 2 classes
    Xp = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # 3 rows, 3 classes
    h, w = 3, 3

    # Rulebook: class 0 -> set to 5, class 1 -> set to 7, class 2 -> set to 9
    rulebook = {
        0: Rule(action="set_color", params={"color": 5}),
        1: Rule(action="set_color", params={"color": 7}),
        2: Rule(action="set_color", params={"color": 9}),
    }

    row_blocks, col_blocks = build_row_col_blocks(labels, h, w)
    aux = AuxData(row_blocks=row_blocks, col_blocks=col_blocks)

    result = glue_once(Xp, labels, rulebook, aux)

    # Row 0 should be all 5, row 1 all 7, row 2 all 9
    if (result[0] == [5, 5, 5] and
        result[1] == [7, 7, 7] and
        result[2] == [9, 9, 9]):
        passed += 1
    else:
        print(f"✗ glue_once: composition failed")
        print(f"  Got: {result}")
        failed += 1

    print(f"glue_once tests: {passed} passed, {failed} failed\n")
    return passed, failed


def main():
    print("\n" + "=" * 60)
    print("M5a VERIFICATION: Action Execution Engine")
    print("=" * 60 + "\n")

    total_passed = 0
    total_failed = 0

    # Run all tests
    p, f = test_class_masks()
    total_passed += p
    total_failed += f

    p, f = test_build_row_col_blocks()
    total_passed += p
    total_failed += f

    p, f = test_build_lut_from_evidence()
    total_passed += p
    total_failed += f

    p, f = test_apply_action()
    total_passed += p
    total_failed += f

    p, f = test_glue_once()
    total_passed += p
    total_failed += f

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print()

    if total_failed == 0:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print(f"✗ {total_failed} CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
