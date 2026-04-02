# Potentials example

Evaluates and compares the three pair potentials (Lennard-Jones,
ten Wolde-Frenkel, Wang-Ramirez-Dobnikar-Frenkel) via the variant-based
`physics::potentials` API: `energy()`, `attractive()`, `repulsive()`,
`hard_sphere_diameter()`, `vdw_integral()`.

With matplotlib enabled, produces comparison plots and WCA decomposition.

## Build and run

```bash
make run
```

## Output

- `exports/potentials_comparison.png` — potential comparison
- `exports/perturbation_lj.png` — WCA decomposition
- `exports/potential_lj.png` — LJ with d_HS marker
