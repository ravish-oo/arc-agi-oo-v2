#!/usr/bin/env python3
"""
M5 Debug: Run a single task with FY detailed diagnostics.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable FY debug logging BEFORE importing
import src.fy as fy
fy.FY_DEBUG_LOG = True
fy.FY_DEBUG_TASK_ID = "025d127b"
fy.FY_DEBUG_THETA_KIND = "identity"

from src.solve import solve_task


def load_single_task(data_dir: Path, task_id: str):
    """Load a single task by ID."""
    challenges_path = data_dir / "arc-agi_training_challenges.json"

    if not challenges_path.exists():
        print(f"ERROR: {challenges_path} not found")
        sys.exit(1)

    with open(challenges_path, 'r') as f:
        data = json.load(f)

    if task_id not in data:
        print(f"ERROR: Task {task_id} not found")
        sys.exit(1)

    task = data[task_id]
    train_pairs = task.get('train', [])
    trains = [(p['input'], p['output']) for p in train_pairs]
    tests = [p['input'] for p in task.get('test', [])]

    return trains, tests


def main():
    parser = argparse.ArgumentParser(description='M5 Debug: Single task with FY diagnostics')
    parser.add_argument('--escalate_policy', type=str, default=None, choices=['E8', '2WL'],
                       help='Escalation policy for Φ refinement')
    args = parser.parse_args()

    task_id = "025d127b"
    data_dir = Path("data")

    escalate_str = f" with escalate_policy={args.escalate_policy}" if args.escalate_policy else ""
    print("=" * 70)
    print(f"M5 DEBUG: Task {task_id} with detailed FY diagnostics{escalate_str}")
    print("=" * 70)
    print()

    trains, tests = load_single_task(data_dir, task_id)

    print(f"Train pairs: {len(trains)}")
    print(f"Test inputs: {len(tests)}")
    print()

    # Run solve_task with diagnostics enabled
    result = solve_task(
        trains,
        tests,
        escalate_policy=args.escalate_policy,
        use_task_color_canon=False,
        lut_density_tau=0.8
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"ok: {result.ok}")
    if result.ok:
        print(f"theta: {result.theta.kind if result.theta else '?'}")
        print(f"rulebook_size: {len(result.rulebook)}")
    else:
        print(f"witness: {result.witness}")

    return 0 if result.ok else 1


if __name__ == '__main__':
    sys.exit(main())
