#!/usr/bin/env python3
"""
M4 Verifier: Check Actions library.

Tests all action functions on toy grids and real samples:
1. Determinism (run twice, identical output)
2. Mask respect (no changes outside mask)
3. Semantics (action-specific checks)
"""

import sys
import json
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.actions.mirror import mirror_h, mirror_v, mirror_diag
from src.actions.shift import shift
from src.actions.rowcol import reorder_rows_by_blocks, reorder_cols_by_blocks, sort_rows_lex, sort_cols_lex
from src.actions.constructors import draw_box_on_components, draw_line_axis_aligned
from src.actions.lut import lut_rewrite
from src.actions.constant import set_color


def check_determinism(action, grid, mask, **params):
    """Check that action is deterministic."""
    result1 = action(grid, mask, **params)
    result2 = action(grid, mask, **params)
    return result1 == result2


def check_mask_respect(original, result, mask):
    """Check that no changes occurred outside mask."""
    h = len(original)
    w = len(original[0]) if original else 0

    for r in range(h):
        for c in range(w):
            if not mask[r][c]:
                if original[r][c] != result[r][c]:
                    return False
    return True


def run_mirror_tests():
    """Test mirror actions."""
    print("=" * 60)
    print("TESTING MIRROR ACTIONS")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: mirror_h on simple grid
    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mask = [[True, True, True], [True, True, True], [True, True, True]]

    # Determinism
    if not check_determinism(mirror_h, grid, mask):
        print("✗ mirror_h: determinism failed")
        failed += 1
    else:
        passed += 1

    # Mask respect
    result = mirror_h(grid, mask)
    if not check_mask_respect(grid, result, mask):
        print("✗ mirror_h: mask respect failed")
        failed += 1
    else:
        passed += 1

    # Idempotence (mirror twice = identity on masked area)
    result1 = mirror_h(grid, mask)
    result2 = mirror_h(result1, mask)

    # Check masked area is restored
    restored = True
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if mask[r][c] and grid[r][c] != result2[r][c]:
                restored = False

    if not restored:
        print("✗ mirror_h: idempotence failed")
        failed += 1
    else:
        passed += 1

    # Test mirror_v
    if not check_determinism(mirror_v, grid, mask):
        print("✗ mirror_v: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test mirror_diag (square mask required)
    if not check_determinism(mirror_diag, grid, mask):
        print("✗ mirror_diag: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test mirror_diag with non-square bbox (should raise)
    rect_mask = [[True, True, True, True], [True, True, True, True], [True, True, True, True]]
    rect_grid = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]
    try:
        mirror_diag(rect_grid, rect_mask)
        print("✗ mirror_diag: should raise on non-square bbox")
        failed += 1
    except ValueError:
        passed += 1

    print(f"Mirror tests: {passed} passed, {failed} failed\n")
    return passed, failed


def run_shift_tests():
    """Test shift action."""
    print("=" * 60)
    print("TESTING SHIFT ACTION")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mask = [[True, True, True], [True, True, True], [True, True, True]]

    # Determinism
    if not check_determinism(shift, grid, mask, dr=1, dc=0):
        print("✗ shift: determinism failed")
        failed += 1
    else:
        passed += 1

    # Mask respect with partial mask
    partial_mask = [[True, True, False], [True, True, False], [False, False, False]]
    result = shift(grid, partial_mask, dr=0, dc=1)
    if not check_mask_respect(grid, result, partial_mask):
        print("✗ shift: mask respect failed")
        failed += 1
    else:
        passed += 1

    # Vacated cells should be 0
    result = shift(grid, mask, dr=1, dc=0)
    if result[0] != [0, 0, 0]:  # Top row should be vacated
        print("✗ shift: vacated cells not zero")
        failed += 1
    else:
        passed += 1

    print(f"Shift tests: {passed} passed, {failed} failed\n")
    return passed, failed


def run_rowcol_tests():
    """Test row/col reordering actions."""
    print("=" * 60)
    print("TESTING ROW/COL ACTIONS")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[3, 2, 1], [1, 2, 3], [2, 1, 3]]
    mask = [[True, True, True], [True, True, True], [True, True, True]]

    # Test sort_rows_lex determinism
    if not check_determinism(sort_rows_lex, grid, mask):
        print("✗ sort_rows_lex: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test sort_cols_lex determinism
    if not check_determinism(sort_cols_lex, grid, mask):
        print("✗ sort_cols_lex: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test reorder_rows_by_blocks
    row_blocks = [[0, 2], [1]]
    if not check_determinism(reorder_rows_by_blocks, grid, mask, row_blocks=row_blocks):
        print("✗ reorder_rows_by_blocks: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test reorder_cols_by_blocks
    col_blocks = [[0, 1], [2]]
    if not check_determinism(reorder_cols_by_blocks, grid, mask, col_blocks=col_blocks):
        print("✗ reorder_cols_by_blocks: determinism failed")
        failed += 1
    else:
        passed += 1

    print(f"Row/col tests: {passed} passed, {failed} failed\n")
    return passed, failed


def run_constructor_tests():
    """Test constructor actions."""
    print("=" * 60)
    print("TESTING CONSTRUCTOR ACTIONS")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[0, 0, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]]
    mask = [[False, False, False, False, False], [False, True, True, True, False],
            [False, True, True, True, False], [False, True, True, True, False],
            [False, False, False, False, False]]

    # Test draw_box_on_components determinism
    if not check_determinism(draw_box_on_components, grid, mask, thickness=1, color=5):
        print("✗ draw_box_on_components: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test draw_line_axis_aligned determinism
    if not check_determinism(draw_line_axis_aligned, grid, mask, axis="row", color=7):
        print("✗ draw_line_axis_aligned: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test invalid color
    try:
        draw_line_axis_aligned(grid, mask, axis="row", color=10)
        print("✗ draw_line_axis_aligned: should raise on invalid color")
        failed += 1
    except ValueError:
        passed += 1

    # Test invalid axis
    try:
        draw_line_axis_aligned(grid, mask, axis="diagonal", color=5)
        print("✗ draw_line_axis_aligned: should raise on invalid axis")
        failed += 1
    except ValueError:
        passed += 1

    print(f"Constructor tests: {passed} passed, {failed} failed\n")
    return passed, failed


def run_lut_tests():
    """Test LUT rewrite action."""
    print("=" * 60)
    print("TESTING LUT ACTION")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mask = [[True, True, True], [True, True, True], [True, True, True]]

    # Simple LUT with r=2
    # We won't define any keys, so everything should remain unchanged
    lut = {}
    if not check_determinism(lut_rewrite, grid, mask, r=2, key_to_color=lut):
        print("✗ lut_rewrite: determinism failed")
        failed += 1
    else:
        passed += 1

    # Test with empty LUT - should leave grid unchanged
    result = lut_rewrite(grid, mask, r=2, key_to_color=lut)
    if result != grid:
        print("✗ lut_rewrite: empty LUT should leave grid unchanged")
        failed += 1
    else:
        passed += 1

    # Test invalid radius
    try:
        lut_rewrite(grid, mask, r=5, key_to_color=lut)
        print("✗ lut_rewrite: should raise on invalid radius")
        failed += 1
    except ValueError:
        passed += 1

    # Test invalid color in LUT
    bad_lut = {tuple([tuple([0, 0, 0, 0, 0])] * 5): 10}
    try:
        lut_rewrite(grid, mask, r=2, key_to_color=bad_lut)
        print("✗ lut_rewrite: should raise on invalid color in LUT")
        failed += 1
    except ValueError:
        passed += 1

    print(f"LUT tests: {passed} passed, {failed} failed\n")
    return passed, failed


def run_constant_tests():
    """Test constant action."""
    print("=" * 60)
    print("TESTING CONSTANT ACTION")
    print("=" * 60)

    passed = 0
    failed = 0

    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mask = [[True, False, True], [False, True, False], [True, False, True]]

    # Determinism
    if not check_determinism(set_color, grid, mask, color=5):
        print("✗ set_color: determinism failed")
        failed += 1
    else:
        passed += 1

    # Mask respect
    result = set_color(grid, mask, color=5)
    if not check_mask_respect(grid, result, mask):
        print("✗ set_color: mask respect failed")
        failed += 1
    else:
        passed += 1

    # Check masked cells are set
    all_set = True
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if mask[r][c] and result[r][c] != 5:
                all_set = False

    if not all_set:
        print("✗ set_color: not all masked cells set to color")
        failed += 1
    else:
        passed += 1

    # Test invalid color
    try:
        set_color(grid, mask, color=15)
        print("✗ set_color: should raise on invalid color")
        failed += 1
    except ValueError:
        passed += 1

    print(f"Constant tests: {passed} passed, {failed} failed\n")
    return passed, failed


def main():
    print("\n" + "=" * 60)
    print("M4 VERIFICATION: Actions Library")
    print("=" * 60 + "\n")

    total_passed = 0
    total_failed = 0

    # Run all tests
    p, f = run_mirror_tests()
    total_passed += p
    total_failed += f

    p, f = run_shift_tests()
    total_passed += p
    total_failed += f

    p, f = run_rowcol_tests()
    total_passed += p
    total_failed += f

    p, f = run_constructor_tests()
    total_passed += p
    total_failed += f

    p, f = run_lut_tests()
    total_passed += p
    total_failed += f

    p, f = run_constant_tests()
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
