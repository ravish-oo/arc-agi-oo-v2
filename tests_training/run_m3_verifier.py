#!/usr/bin/env python3
"""
M3 Verifier: Check Φ (input-only WL partition).

Checks on training data (inputs only):
1. Determinism: running Φ twice yields identical labels
2. Presentation invariance: class histograms match across all M1 transforms
3. No escalation used (M3 baseline)
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pi_orient import canon_orient, apply_transform, TRANSFORMS
from src.phi_rel import build_rel_structure
from src.phi_wl import wl_refine


# Fallback toy grids if no ARC data
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


def load_training_inputs(data_dir: Path):
    """
    Load training input grids only (no Y).
    Returns list of grids.
    """
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print("ARC data not found, using toy grids\n")
        return TOY_GRIDS

    print(f"Loading grids from {challenges_path}")
    with open(challenges_path, 'r') as f:
        data = json.load(f)

    grids = []
    for task_id, task in data.items():
        # Collect training inputs only
        for pair in task.get('train', []):
            if 'input' in pair:
                grids.append(pair['input'])

    print(f"Loaded {len(grids)} training input grids\n")
    return grids


def check_determinism(grids):
    """
    Check: running Φ twice on same canonized grid yields identical labels.
    """
    print("=" * 60)
    print("CHECKING DETERMINISM")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    for i, X in enumerate(grids):
        # Canonize
        Xc = canon_orient(X).grid

        # Run Φ twice
        rel = build_rel_structure(Xc)
        labels1, iters1, esc1 = wl_refine(rel)
        labels2, iters2, esc2 = wl_refine(rel)

        if labels1 == labels2 and iters1 == iters2 and esc1 == esc2:
            passed += 1
        else:
            failed += 1
            failures.append({
                'grid_idx': i,
                'labels_match': labels1 == labels2,
                'iters_match': iters1 == iters2,
                'esc_match': esc1 == esc2,
            })

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    if failures:
        print("FAILURES:")
        for f in failures[:5]:
            print(f"  Grid {f['grid_idx']}:")
            print(f"    Labels match: {f['labels_match']}")
            print(f"    Iters match: {f['iters_match']}")
            print(f"    Escalation match: {f['esc_match']}")
        print()

    return passed, failed


def check_presentation_invariance(grids):
    """
    Check: class histograms match across all M1 transforms of the same grid.
    """
    print("=" * 60)
    print("CHECKING PRESENTATION INVARIANCE")
    print("=" * 60)

    passed = 0
    failed = 0
    failures = []

    for i, X in enumerate(grids):
        # Baseline: canon(X)
        Xc = canon_orient(X).grid
        rel = build_rel_structure(Xc)
        labels_base, _, _ = wl_refine(rel)
        hist_base = tuple(sorted(Counter(labels_base).values()))
        num_classes_base = len(set(labels_base))

        # Check all transforms
        invariant = True
        for code in TRANSFORMS:
            # Transform, then canon
            Xt = apply_transform(X, code)
            Xtc = canon_orient(Xt).grid

            # Run Φ
            rel_t = build_rel_structure(Xtc)
            labels_t, _, _ = wl_refine(rel_t)
            hist_t = tuple(sorted(Counter(labels_t).values()))
            num_classes_t = len(set(labels_t))

            # Check histogram and class count match
            if hist_t != hist_base or num_classes_t != num_classes_base:
                invariant = False
                failures.append({
                    'grid_idx': i,
                    'transform': code,
                    'hist_base': hist_base,
                    'hist_transform': hist_t,
                    'num_classes_base': num_classes_base,
                    'num_classes_transform': num_classes_t,
                })
                break

        if invariant:
            passed += 1
        else:
            failed += 1

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    if failures:
        print("FAILURES:")
        for f in failures[:5]:
            print(f"  Grid {f['grid_idx']}, transform {f['transform']}:")
            print(f"    Base histogram: {f['hist_base']}")
            print(f"    Transform histogram: {f['hist_transform']}")
            print(f"    Base classes: {f['num_classes_base']}")
            print(f"    Transform classes: {f['num_classes_transform']}")
        print()

    return passed, failed


def check_no_escalation(grids):
    """
    Check: escalation is never used in M3 baseline.
    """
    print("=" * 60)
    print("CHECKING NO ESCALATION (M3 baseline)")
    print("=" * 60)

    passed = 0
    failed = 0

    for i, X in enumerate(grids):
        Xc = canon_orient(X).grid
        rel = build_rel_structure(Xc)
        _, _, used_esc = wl_refine(rel)

        if not used_esc:
            passed += 1
        else:
            failed += 1

    print(f"Passed: {passed}/{len(grids)}")
    print(f"Failed: {failed}/{len(grids)}\n")

    return passed, failed


def compute_statistics(grids):
    """
    Compute and display statistics.
    """
    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)

    class_counts = []
    samples = []

    for i, X in enumerate(grids):
        Xc = canon_orient(X).grid
        rel = build_rel_structure(Xc)
        labels, iters, _ = wl_refine(rel)

        num_classes = len(set(labels))
        class_counts.append(num_classes)

        # Collect samples
        if i < 3:
            h, w = len(Xc), len(Xc[0]) if Xc else 0
            first_10 = labels[:10] if len(labels) >= 10 else labels
            samples.append({
                'idx': i,
                'shape': (h, w),
                'num_classes': num_classes,
                'first_10_labels': first_10,
                'iters': iters,
            })

    if class_counts:
        avg_classes = sum(class_counts) / len(class_counts)
        min_classes = min(class_counts)
        max_classes = max(class_counts)

        print(f"Average classes per grid: {avg_classes:.2f}")
        print(f"Min classes: {min_classes}")
        print(f"Max classes: {max_classes}")
        print()

        print("Sample grids (first 3):")
        for s in samples:
            print(f"  Grid {s['idx']}: shape={s['shape']}, classes={s['num_classes']}, iters={s['iters']}")
            print(f"    First 10 labels: {s['first_10_labels']}")
        print()


def main():
    print("\n" + "=" * 60)
    print("M3 VERIFICATION: Φ (Input-Only WL Partition)")
    print("=" * 60 + "\n")

    # Load data
    data_dir = Path(__file__).parent.parent / "data"
    grids = load_training_inputs(data_dir)
    print(f"Processing {len(grids)} grids\n")

    # Run checks
    det_pass, det_fail = check_determinism(grids)
    inv_pass, inv_fail = check_presentation_invariance(grids)
    esc_pass, esc_fail = check_no_escalation(grids)

    # Statistics
    compute_statistics(grids)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Grids processed: {len(grids)}")
    print()
    print(f"Determinism:               {det_pass} pass, {det_fail} fail")
    print(f"Presentation invariance:   {inv_pass} pass, {inv_fail} fail")
    print(f"No escalation (M3):        {esc_pass} pass, {esc_fail} fail")
    print()

    total_fails = det_fail + inv_fail + esc_fail

    if total_fails == 0:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print(f"✗ {total_fails} CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
