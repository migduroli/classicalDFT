# Solver example

Computes the full LJ fluid phase diagram using the mean-field DFT framework:
pressure isotherms at several temperatures, spinodal and coexistence curves
via `find_spinodal()` and `find_coexistence()`, and critical point estimation.

Uses the declarative `Species` + `Interaction` + `make_bulk_weights()` API
with `WhiteBearII` FMT and WCA splitting.

With matplotlib enabled, produces isotherm and phase diagram plots.

## Build and run

```bash
make run
```

## Output

- `exports/isotherms.png` — pressure isotherms
- `exports/phase_diagram.png` — coexistence and spinodal curves
