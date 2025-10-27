# Start here: lean repo scaffold (exact files)

```
src/
  solve.py                   # pipeline entry
  pi_orient.py               # Π
  p_menu.py                  # P (feasibility)
  phi_rel.py                 # build relations for Φ
  phi_wl.py                  # 1-WL (+ single escalation choice hardcoded)
  actions/
    mirror.py
    shift.py
    rowcol.py                # row/col perms & sorts induced from Φ
    constructors.py          # draw_box, draw_line
    lut.py                   # LUT-r with local OFA
    constant.py
  fy.py                      # FY driver (per-class learning)
  glue.py                    # single stitch
  utils.py                   # stable 64-bit hash, row-major helpers
  datasets.py                # load ARC JSON
tests_training/
  run_training_verifier.py   # runs on all training tasks and prints a table
docs/
  normal_form.md             # your anchor (paste it)
```

No other docs. No constants doc. Fixed orders live in code (e.g., `TRANSFORMS = [...]`, `ACTION_MENU = [...]`).
