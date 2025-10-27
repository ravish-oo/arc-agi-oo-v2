"""
Φ Weisfeiler-Leman refinement (1-WL, input-only, deterministic).

Runs color refinement over multi-relational graph to produce stable pixel partition.
No absolute coordinates in labels; deterministic signature mapping.
"""

from typing import List, Dict, Tuple, Optional
from collections import Counter
from src.phi_rel import RelStructure, RelationTag

# Global counter to track Φ calls (for M6 "No Φ after GLUE" verification)
PHI_CALL_COUNTER = 0


def wl_refine(rel: RelStructure,
              max_iters: int = 20,
              escalate: Optional[str] = None) -> Tuple[List[int], int, bool]:
    """
    1-WL refinement on multi-relational graph.

    Args:
        rel: RelStructure from build_rel_structure
        max_iters: maximum refinement iterations (default 20)
        escalate: escalation mode (unused in M3, hook for M5)

    Returns:
        (labels, iters, used_escalation)
        - labels: stable integer class id per node (len n), starting at 0..C-1
        - iters: number of refinement iterations executed
        - used_escalation: always False in M3 (hook present for M5)

    Algorithm:
        1. Initialize labels from colors
        2. Fixed tag order: ["E4", "R_row", "R_col"] + sorted(R_cc_k tags)
        3. Each iteration:
           - For each node, build signature from (current_label, neighbor_multisets)
           - Deterministically map signatures to new labels (sort unique sigs)
        4. Stop when labels stabilize or max_iters

    Deterministic; no randomness; no absolute coordinates in labels.
    """
    global PHI_CALL_COUNTER
    PHI_CALL_COUNTER += 1

    n = rel.n

    if n == 0:
        return ([], 0, False)

    # Initialize labels from colors
    labels = rel.colors[:]

    # Determine fixed tag order
    tag_order = _get_tag_order(rel)

    # Refinement loop
    iters = 0
    for iteration in range(max_iters):
        iters = iteration + 1

        # Build signatures for all nodes
        signatures = []
        for i in range(n):
            sig = _build_signature(i, labels, rel, tag_order)
            signatures.append(sig)

        # Map signatures to new labels deterministically
        new_labels = _signatures_to_labels(signatures)

        # Check if stable
        if new_labels == labels:
            break

        labels = new_labels

    # Escalation hook (disabled in M3)
    used_escalation = False

    return (labels, iters, used_escalation)


def _get_tag_order(rel: RelStructure) -> List[RelationTag]:
    """
    Determine fixed tag order for deterministic refinement.

    Order: ["E4", "R_row", "R_col"] + sorted(R_cc_k tags)
    """
    tag_order = ["E4", "R_row", "R_col"]

    # Add R_cc_k tags in sorted order
    cc_tags = sorted([tag for tag in rel.neigh.keys() if tag.startswith("R_cc_")])
    tag_order.extend(cc_tags)

    return tag_order


def _build_signature(i: int, labels: List[int], rel: RelStructure,
                     tag_order: List[RelationTag]) -> Tuple:
    """
    Build signature for node i.

    Signature: (current_label, multiset_tag1, multiset_tag2, ...)
    where each multiset is encoded as a sorted tuple of neighbor labels.
    """
    sig_parts = [labels[i]]

    for tag in tag_order:
        if tag not in rel.neigh:
            # Tag not present (e.g., R_cc_k for missing color)
            sig_parts.append(())
            continue

        # Get neighbor labels for this tag
        neighbor_labels = []
        for j in rel.neigh[tag][i]:
            neighbor_labels.append(labels[j])

        # Encode multiset as sorted tuple (canonical form)
        multiset = tuple(sorted(neighbor_labels))
        sig_parts.append(multiset)

    return tuple(sig_parts)


def _signatures_to_labels(signatures: List[Tuple]) -> List[int]:
    """
    Map signatures to integer labels deterministically.

    Sort unique signatures and assign labels 0..C-1 in that order.
    """
    # Get unique signatures and sort them
    unique_sigs = sorted(set(signatures))

    # Create mapping: signature -> label
    sig_to_label = {sig: label for label, sig in enumerate(unique_sigs)}

    # Map each node's signature to its label
    labels = [sig_to_label[sig] for sig in signatures]

    return labels
