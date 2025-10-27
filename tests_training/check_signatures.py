#!/usr/bin/env python3
"""
Check function signatures for M2 implementation.
Verifies that enumerate_feasible_P has the correct signature.
"""

import ast
import sys
from pathlib import Path


def check_signatures():
    """Check that M2 functions have correct signatures."""
    source = Path(__file__).parent.parent / "src" / "p_menu.py"

    if not source.exists():
        print(f"ERROR: {source} not found")
        return 1

    tree = ast.parse(source.read_text())

    found_enumerate = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "enumerate_feasible_P":
                found_enumerate = True
                # Check it has 1 argument
                if len(node.args.args) != 1:
                    print(f"ERROR: enumerate_feasible_P should have 1 arg, has {len(node.args.args)}")
                    return 1
                # Check return annotation exists
                if node.returns:
                    print("✓ enumerate_feasible_P signature OK")
                else:
                    print("WARNING: enumerate_feasible_P missing return annotation")

    if not found_enumerate:
        print("ERROR: enumerate_feasible_P not found")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(check_signatures())
