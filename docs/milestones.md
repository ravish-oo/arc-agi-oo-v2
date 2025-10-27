**Must-haves (to ship a working solver):**

1. A single anchor doc: `docs/normal_form.md` (your spec pasted verbatim).
2. Deterministic code skeleton with fixed orders baked into **code** (not a doc).
3. Π (orientation canon) implemented and tested.
4. P (finite feasibility chooser) implemented and tested (feasibility only).
5. Φ (1-WL over input-only relations) implemented and tested.
6. Actions lib (mirror, shift, row/col reorder, constructors, LUT-r, constant).
7. FY driver (per-class, first exact action, ≤1 escalation).
8. GLUE once + final training equality check.
9. A tiny **training verifier** that runs end-to-end on training pairs and prints pass/fail per task.
   That’s your ground truth harness. No extra receipts unless you want them.

**Nice-to-haves (skip for now):**

* Fancy receipts, MDL cost tables doc, witness formatting guidelines, extra reviewers.
  We can add later after baseline works.

---

# Milestones (tight, minimal)

## M1 — Π only ✅ COMPLETE

**Implement:** `canon_orient(Z) -> (Zc, undo)` using fixed order over D8 (+ transpose when shapes differ).
**Verifier (training-only):**

* Idempotence: `canon_orient(canon_orient(Z)).Zc == canon_orient(Z).Zc`
* Round-trip on every training Y: `undo(canon_orient(Y)) == Y`

**Exit when:** both checks 100% on training.

## M2 — P (feasibility only)

**Implement:** `enumerate_feasible_P(trains_pi)`
Include: identity, H/V/180, 90/270 if square, transpose when needed, scale up {2,3}, pool {2,3} with aggregators {majority, first_nonzero, center}, tiling via exact divisors, block perm/subst on exact k×k partitions (color-neutral).

**Verifier:** for every train pair (after Π):

* `Θ_feas` non-empty
* No equality checks to Y in code (simple grep rule)
* Log counts per task so you can spot outliers

**Exit when:** all pairs have ≥1 feasible θ.

## M3 — Φ (input-only 1-WL)

**Implement:** build relations (C_k, E4, R_row, R_col, R_cc,k; optional bands/phases only if detected). Run 1-WL to fixed point. Choose **one** escalation path (either 2-WL or +E8) and hardcode it.

**Verifier:**

* Purity grep: no absolute `(r,c)`, no Y/Δ in `phi_*`
* Presentation check: rotate/flip Π(X) variants → same class histogram and stable 64-bit map hash (compute once per grid)

**Exit when:** all training inputs pass the checks.

## M4 — Actions (library only)

**Implement:** pure functions operating inside a mask.

* mirror, shift, rowcol reorder/sort, draw_box, draw_line, LUT-r (r∈{2,3}, optional r=4 once), constant.

**Verifier:** golden toy cases for each action (tiny synthetic grids).
Keep it small: 1–2 tests per action.

**Exit when:** all goldens pass.

## M5 — FY (per-class learning)

**Implement:** `learn_rules(trains_pi, θ)`:

* Compute X' = Pθ(ΠX), Y' = ΠY
* Φ(X'), collect Δ per class
* Try `ACTION_MENU` in fixed order; accept first that is exact across **all** training evidence for that class
* If conflict, do single escalation then retry; else mark UNSAT with a short message

**Verifier (training-only):**

* For each task, either produce a rulebook (some θ) or produce UNSAT; no partial “almost”
* Zero unseen-key at train time (LUT coverage must be total on evidence)

**Exit when:** a healthy subset of training tasks produce rulebooks; the rest return UNSAT cleanly.

## M6 — GLUE + final training equality

**Implement:** one-pass GLUE on disjoint masks, then undo Π.

**Verifier:** for every task where FY produced a rulebook:

* Training outputs match bit-for-bit
* Determinism: run twice, identical bytes

**Exit when:** matches hold on that subset.

*(Optional later: MDL tie-break among exact θ; not needed to reach first solves.)*

---

# Claude Code setup (prompts you actually use)

**Implementer prompt (one per milestone):**

> You are the Implementer. Only touch files for Milestone M<N>. Follow docs/normal_form.md. No heuristics, no randomness, no new globals. Implement:
>
> * [list functions]
>   Add minimal tests in tests_training/ for this milestone only. Do not modify other modules. Provide short commit message and run tests_training/run_training_verifier.py at the end and paste its summary.

**Reviewer prompt (single, reused):**

> You are the Reviewer. Reject if any of the following is violated:
>
> * P uses equality to Y (grep ‘==’ against Y in p_menu.py).
> * Φ references absolute indices or Y/Δ.
> * More than one WL escalation.
> * Actions operate outside mask.
> * FY accepts an action that is not first that passes.
> * Any non-deterministic iteration over dicts/sets without sorting.
>   Run tests_training/run_training_verifier.py; output must show 100% pass for this milestone’s gates.

Keep it short, mechanical.

---

# Day-1 “do this now” checklist

1. Create repo with the scaffold above.
2. Paste your spec into `docs/normal_form.md`.
3. Ask Claude to complete **M1** exactly.
4. Run the tiny M1 verifier on training Ys.
5. If green, proceed to **M2**. If red, fix M1 before touching anything else.

That’s it. No extra docs, no receipts, no MDL—for now. We get Π→P→Φ→Actions→FY→GLUE working on a slice of training tasks first, entirely verified against the provided training solutions. When you’re ready, I can draft the exact function signatures for M1–M2 so you can paste them in and let Claude fill bodies.
