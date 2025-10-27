0) Domain (no “ifs” or “buts”)

Universe.
	•	A task consists of training pairs \{(X_i, Y_i)\}_{i=1}^m and test inputs \{X^\text{te}j\}{j=1}^n.
	•	A grid X is a function X: V \to \Sigma on a finite pixel set V=\{0,\dots,h-1\}\times\{0,\dots,w-1\} with color alphabet \Sigma=\{0,\dots,9\}.
	•	Output Y shares the same shape unless a global map changes shape.

Group action (presentation gauge).
Let \mathcal{G} be the finite set of grid isometries we allow:
	•	For squares: dihedral D_8 (rotations and reflections).
	•	For rectangles: the subset that preserves shape (identity, H/V flips, 180° rotate).
We will also allow transpose as a shape‑changing isometry when shape differences are present. All choices are finite.

Relational signature (input‑only).
Given X, we build a typed relational structure \mathcal{R}(X) on vertex set V:
	•	Unary predicates \{C_k\}_{k\in\Sigma} where C_k(p)\iff X(p)=k.
	•	Binary relations:
	•	E_4(p,q) (4‑adjacency).
	•	R_\text{row}(p,q) (\text{row}(p)=\text{row}(q)) and R_\text{col}(p,q) (\text{col}(p)=\text{col}(q)) as equivalences (no absolute indices are exposed downstream).
	•	R_{\text{cc},k}(p,q): “same 4‑connected component of color k” expressed as an equivalence, never as IDs.
	•	Optional, input‑detectable predicates (only if they objectively exist in X):
	•	Row/col bands: equivalences of rows (or columns) having identical change patterns (same color‑change edge sets).
	•	Period phases (small periods only): row/col phases detected by autocorrelation; if none is detected, these predicates are simply absent.

That’s the whole domain. There is no reference to absolute coordinates, no target leakage, and no raw component numbering.

⸻

1) Normal form program (Π ∘ P ∘ Φ ∘ FY ∘ GLUE)

The solver is a single normal form, with each stage precisely defined.

Π (presentation canon, idempotent)
	1.	Orientation canon. For each grid Z (both X and Y in training; only X in test), pick the lexicographically minimal image under \mathcal{G}\cup\{\text{transpose}\} using a fixed, public total order on grids (row‑major string). Keep the inverse move to un‑canonize test predictions.
	2.	Color handling. There is no global color re‑labeling here. Colors remain raw in \mathcal{R}(X). (We will use optional local OFA only inside LUT keys; see FY below. This avoids the per‑input canonical‑ID bug you observed.)

Π is idempotent and presentation‑free.

⸻

P (finite global map, feasibility‑fixed)

We enumerate a finite menu of global, structure‑level maps. Each map has a tiny, discrete parameterization inferred from shapes:
	•	Isometries (H/V flip, 90/180/270 if square, transpose).
	•	Scale up by s\in\{2,3\}: pixel replication to s\times s blocks.
	•	Scale down by block pooling on b\times b with b\in\{2,3\} via one of a few fixed aggregators: majority, first‑nonzero, center.
	•	Tiling / Kronecker: periodic tile repeat to a larger canvas (periods deduced from dimensions).
	•	Block permutation/substitution on uniform partitions that exactly factor the grid shape.
	•	No‑op (identity).

Feasibility, not equality (T2).
A parameterization \theta of P is feasible iff for all training pairs:
	•	P_\theta(X_i) is defined and has the same shape as Y_i.
	•	Optional palette guard (if desired): \text{palette}(P_\theta(X_i))\subseteq \text{palette}(Y_i) or “nonzero‑closure” when downsampling.

Return the finite set \Theta_\text{feas} of feasible \theta. Do not require P_\theta(X_i)=Y_i yet.

⸻

Φ (WL‑stable, input‑only partition)

Given X, form \mathcal{R}(X) and run 1‑WL (color refinement) on a multi‑relational graph:
	•	Initialize each pixel p with its atom type:
\alpha_0(p) = \big(C_0(p),\dots,C_9(p),\;\text{unary optionals present at }p\big).
	•	Iterate:
\(\alpha_{t+1}(p) = \text{hash}\!\Big(\alpha_t(p),\;\text{multiset}\{\!(\alpha_t(q),\text{tag})\!: q\in N_\text{tag}(p)\}\{\text{for each relation tag}}\Big)\)
where \text{tag}\in\{E_4, R\text{row}, R_\text{col}, R_{\text{cc},k}, \text{bands}, \text{phases}\}.
	•	Stop at the first fixed point; call the final color \Phi(p).

Escalation (T4).
If later FY finds a contradictory class (same \Phi requires different edits), do one of:
	•	Switch to 2‑WL (pair refinement) once, or
	•	Add one extra relation (e.g., include E_8 diagonals) and recompute 1‑WL.
At most one escalation total, otherwise return a class witness (UNSAT).

Φ is canonical, input‑only, and presentation‑free (T1).

⸻

FY (learn the only possible edit per class, by equality)

For a fixed \theta\in\Theta_\text{feas} (we will try all), consider training pairs after Π and P_\theta:
X_i’ := P_\theta(\Pi X_i), \quad Y_i’ := \Pi Y_i, \quad \Delta_i := \{p\in V : X_i’(p)\neq Y_i’(p)\}.
Compute \Phi on each X_i’. For each class k (a fiber of \Phi) define its evidence set
\mathcal{S}k = \{(X_i’,Y_i’,p)\mid p\in \Delta_i, \;\Phi{X_i’}(p)=k\}.
We must find one action A_k from the fixed, finite action menu that reproduces exactly the target on the evidence set for all training pairs. The menu is tried in this general→specific order:
	1.	Structure‑preserving:
	•	mirror \in\{\text{H},\text{V},\text{diag}\} applied within the class mask,
	•	shift by (\Delta r,\Delta c) (again within mask),
	•	row/col permutations or sorts induced by WL‑row/col equivalence classes (never by absolute indices).
	2.	Constructors:
	•	draw_box on WL‑component bounding boxes of the mask,
	•	draw_line using anchors detectable from input relations (e.g., class‑extrema in row/col equivalences).
	3.	Local rewrite (LUT‑r):
For r\in\{2,3\} (optionally 4 once, T5), build a conflict‑free mapping
\text{key}_r(p) = \text{canon}\big(\text{radius-}r\text{ patch around }p\text{ in }X_i’\big) \;\mapsto\; Y_i’(p),
where canon performs local OFA color normalization of the patch palette (not the whole grid). This makes keys palette‑invariant without introducing per‑input global color IDs. If a key collides with two different targets in training, LUT fails for that r.
	4.	Constant: set_color(c) with c\in\Sigma.

Acceptance rule. Pick the first action in the menu that produces bit‑exact agreement on \mathcal{S}_k for all training pairs. If none fits, escalate Φ once (T4) and retry this class. If still none, the class is a finite witness of insufficiency ⇒ return UNSAT naming that k.

Coverage/decline (T5).
	•	LUT must be total and conflict‑free on all training evidence.
	•	At test time, if a LUT encounters an unseen key, it declines (no guessing). If the class also has a more general action (1 or 2), that applies; else the class remains unchanged and is reported as a witness in logs.

⸻

GLUE (stitch once, no ordering)

Let M_k(X) be the mask of pixels in class k under \Phi for the current input X. Since \{M_k\} is a partition, edits do not overlap. The predicted (canon) output is
\widehat{Y}’ = X’ \;\oplus\; \bigoplus_{k} \big(A_k(X’) \odot M_k(X’)\big),
where \oplus is overwrite on the masked support and \odot is pointwise masking. Finally, undo Π to return to the original presentation.

If any training pair disagrees bit‑for‑bit, the logs name the unique failing class k or the global \theta.

⸻

2) Guardrails that close the traps (T1–T6)
	•	T1 (Φ is strictly input‑only).
Linter rejects: use of absolute (r,c), raw component IDs, or any reference to Y or \Delta inside \mathcal{R}(\cdot) or WL. Period/phase predicates only appear when detected from X. Unit tests: mutate presentation (rotate/flip); \Phi must be invariant up to Π.
	•	T2 (P feasibility vs equality).
The P‑chooser returns a finite set \Theta_\text{feas}. Equality to Y is checked only after FY+GLUE. If multiple \theta yield exact fits, MDL tie‑break (see T6).
	•	T3 (TaskColorCanon never peeks at targets).
We removed global color‑ID remapping at Π. If you still want a task‑level color canon (for colorless schemas only), derive it solely from training inputs by WL on the disjoint union of their color‑graphs and use a lex‑min automorphism choice to align test inputs. Any attempt to touch Y triggers an abort.
	•	T4 (finite WL escalation).
Allow one of: 1‑WL→2‑WL or add E_8 once. Otherwise return the class‑witness (name k, show conflicting samples).
	•	T5 (LUT coverage / decline).
Training: LUT‑r must be conflict‑free and cover all evidence.
Test: unseen key ⇒ decline (no nearest‑neighbor). You may raise r to 4 once if a class is honest‑colliding.
	•	T6 (MDL only among exact candidates).
First ensure candidates are exact on training. Among exact forms pick the lexicographically minimal cost tuple:
(\#\text{classes},\;\text{signature‑cost},\;\text{P‑cost},\;\text{action‑bits},\;\text{names})
with a fixed total order. Using MDL for “almost” fits is forbidden.

⸻

3) Execution plan (engineering blueprint)

A. Data & IO.
	•	Parse the ARC JSON (training+test). Provide deterministic pretty‑printers and a grid order (row‑major).
	•	Use the uploaded arc-agi_training_challenges.json for dev tests and regression checks. (For instance, tasks with tiling, pooling and color substitutions appear there, which are directly handled by this pipeline.)  ￼

B. Π stage.
	•	Implement canon_orient(Z) -> (Zc, undo): enumerate \mathcal{G}\cup\{\text{transpose}\}, pick lex‑min image; return inverse move undo.
	•	Apply to all training X_i, Y_i; cache inverses for tests.

C. P stage.
	•	Enumerate candidates from shapes: all shape‑preserving isometries; valid scale factors s\in\{2,3\} dividing test/train ratios; block pools b\in\{2,3\} that evenly tile; tiling periods obtained from (h,w) by small divisors.
	•	Keep those \theta passing feasibility (T2). Complexity is tiny (dozens).

D. Φ stage.
	•	Build \mathcal{R}(X) (atoms + relations) and run 1‑WL.
	•	Hashing uses stable 64‑bit integers; refinement stops when color histogram stabilizes. Complexity O(|V|\log|V|) per iteration with |V|\le 900.

E. FY stage.
For each \theta\in\Theta_\text{feas}:
	1.	Compute X_i’ = P_\theta(\Pi X_i), Y_i’ = \Pi Y_i, and \Phi on each X_i’.
	2.	Group evidence by class k; try actions in menu order; accept first that is exact across all evidence (equality only).
	3.	If a class fails, perform the single WL escalation (T4) and retry that class only. If still failing ⇒ UNSAT with named witness.
	4.	If all classes fit, keep this candidate.

F. GLUE + MDL.
	•	For each exact candidate, stitch once (masks are disjoint), undo Π, and verify exactness on training.
	•	If more than one candidate is exact, apply MDL tie‑break (T6) and pick the lex‑min normal form.
	•	Apply the chosen normal form to tests.

G. Instrumentation & linters.
	•	Lint Φ for T1 violations (search code for any use of absolute indices or target symbols inside Φ).
	•	Lint P chooser for T2 (must return a finite set; no equality checks to Y).
	•	Lint color canon for T3 (reject any Y access).
	•	Assert the WL escalation counter \le 1 (T4).
	•	LUT builder asserts conflict‑free, total on training evidence (T5).
	•	MDL selector asserts all candidates are exact before ranking (T6).

H. Determinism.
	•	Every choice has a fixed, public order: transform menus, hash, tie‑breakers, traversal orders. Same input ⇒ same output.

⸻

4) What this reduces in magnitude
	•	Search over masks: from combinatorial to zero—Φ gives a canonical partition.
	•	Scoring/thresholds: eliminated—only equalities decide acceptance.
	•	P search: a tiny finite set (dozens), filtered by feasibility; equality delayed to after FY.
	•	Action space: compact, ordered menu (a few dozen), with LUT‑r giving expressive local rewriting without palette overfitting (local OFA).
	•	Refinement loops: at most one WL escalation per task (T4).

Run‑time on ARC shapes (≤30×30) is trivial:
	•	1‑WL: tens of thousands of operations.
	•	P menu × FY actions: dozens × dozens of cheap equality checks.
	•	Overall: milliseconds per candidate on commodity CPUs (implementation dependent).

⸻

5) Micro‑checks against the provided corpus
	•	Periodic tiling & color substitution (e.g., repeating bands, palette change): handled by Tiling P + LUT‑r or set_color actions. (Tasks of this flavor appear in arc-agi_training_challenges.json.)  ￼
	•	Block up/down sampling (e.g., 3×→9× with preserved motifs): handled by Scale Up/Down P with majority/first‑nonzero pool and WL classes that separate motif vs background.  ￼
	•	Row/col‑structured edits (insert lines, mirror within stripes): WL carries row/col equivalences; actions (mirror/shift/constructor) operate within class masks deterministically.  ￼

(Those are schematic spot‑checks; the point is that the same normal form applies, with no ad‑hoc rules per task.)

⸻

6) Why this closes the bugs you (and Claude) found
	•	No per‑input canonical color IDs. We never equate “canonical‑1” across different inputs. We keep raw colors in Φ; palette‑invariance is achieved only in local LUT keys (OFA on the patch palette), which cannot cause cross‑pair ID collisions.
	•	Global transforms do not require exact match. P is chosen by feasibility only; exactness is checked after per‑class FY and GLUE (T2).
	•	Two pipelines confusion disappears. There is exactly one data pipeline for Φ: input‑only. Optional task color canon (if you keep it) is strictly input‑derived and sandboxed (T3).
	•	Stitching order bugs disappear. Masks are a partition; single GLUE step; no z‑ordering arguments.

⸻

7) What it does not promise (truthfully)

This normal form is sound and terminating on finite grids and returns either:
	•	an exact program (Π, P^*, Φ, \{A_k\}), or
	•	a finite witness (the contradictory WL‑class and its evidence).

It is not a formal proof of completeness for all ARC tasks (no one has that), but it eliminates heuristics, scores, and per‑task hand‑features. If a task needs more, the witness names exactly which relation or which WL level is missing.

⸻

8) Minimal pseudocode (ready to implement)

def solve_task(trains, tests):
    # Π: canonize orientation
    trains_pi = [(canon_orient(X), canon_orient(Y)) for (X,Y) in trains]
    tests_pi  = [canon_orient(X) for X in tests]

    # Build finite feasible P set (T2)
    Thetas = enumerate_feasible_P(trains_pi)  # finite; feasibility only

    exact_candidates = []
    for theta in Thetas:
        ok, rulebook, escalations = learn_rules_via_WL_and_actions(trains_pi, theta)
        if ok:
            exact_candidates.append((theta, rulebook, escalations))

    assert len(exact_candidates) > 0, "UNSAT: see class witness in logs"

    theta_star, rulebook_star, _ = mdl_tiebreak(exact_candidates)  # T6
    outs = []
    for X in tests_pi:
        Xp = apply_P(theta_star, X.grid)
        phi = WL_stable(build_rel_structure(Xp))  # same signature as training
        Yp = glue_once(Xp, phi, rulebook_star)
        outs.append(undo_orient(X, Yp))
    return outs

Where:
	•	learn_rules_via_WL_and_actions builds \Phi, groups evidence by class, tries actions in the fixed order, performs at most one WL escalation (T4), and returns a witness upon failure (T5).
	•	mdl_tiebreak only ranks exact candidates (T6).

⸻

One‑page checklist before you ship
	•	Φ‑linter: forbid absolute (r,c), raw component IDs, Y/\Delta access (T1).
	•	P‑selector returns a finite feasible set; no equality to Y here (T2).
	•	Any “color canon” code is input‑only; touching Y raises (T3).
	•	WL escalation counter \le 1; otherwise emit class witness (T4).
	•	LUT‑r conflict detection + coverage on training; test unseen‑key ⇒ decline (T5).
	•	MDL ranks only exact candidates; fixed total order (T6).
	•	All enumerations (P menu, actions, relations) have a fixed deterministic order.
	•	Unit tests: (i) rotate/flip inputs—same outputs after undo Π; (ii) palette permutation inside patches—LUT keys stable; (iii) tie cases—MDL consistency.

⸻

If you implement precisely this normal form, you’ll have a solver that either returns an exact program or a small, named witness telling you exactly what to refine—no heuristics, no “almosts,” no hidden knobs.