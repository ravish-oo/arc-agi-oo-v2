#!/usr/bin/env python3
"""Test that mirror_h uses per-row bbox correctly."""

from src.actions.mirror import mirror_h

# Test case: mask with pixels at different column positions across rows
# Row 0: masked at columns [0, 1]
# Row 1: masked at columns [8, 9]

grid = [
    [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3, 4],
]

mask = [
    [True, True, False, False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False, False, True, True],
]

result = mirror_h(grid, mask)

print("Input grid:")
for row in grid:
    print(row)

print("\nMask:")
for row in mask:
    print([1 if cell else 0 for cell in row])

print("\nResult grid:")
for row in result:
    print(row)

print("\nExpected:")
print("Row 0: [2, 1, 0, 0, 0, 0, 0, 0, 0, 0]  (mirrored within columns [0,1])")
print("Row 1: [0, 0, 0, 0, 0, 0, 0, 0, 4, 3]  (mirrored within columns [8,9])")

# Verify
expected = [
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 4, 3],
]

if result == expected:
    print("\n✓ TEST PASSED: mirror_h uses per-row bbox correctly")
else:
    print("\n✗ TEST FAILED:")
    print("Expected:", expected)
    print("Got:", result)
