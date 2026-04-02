# Interaction example

Demonstrates the mean-field interaction framework: WCA/BH splitting of the
LJ attractive tail, analytical van der Waals parameter, bulk thermodynamics
f(rho) and mu(rho) via `make_bulk_weights()`, and grid convergence of the
numerical a_vdw integral as a function of grid spacing.

With matplotlib enabled, produces splitting, free energy, and convergence plots.

## Build and run

```bash
make run
```

## Output

- `exports/interaction_wca_bh.png` — WCA vs BH splitting
- `exports/interaction_free_energy.png` — mean-field free energy density
- `exports/interaction_convergence.png` — a_vdw grid convergence
