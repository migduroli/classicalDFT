# FMT example

Compares Fundamental Measure Theory models (Rosenfeld, RSLT, White Bear I/II)
against exact hard-sphere equations of state for bulk excess free energy,
chemical potential, and pressure (via the Gibbs-Duhem consistency check
P/(rho kT) = 1 + rho*(mu_ex - f_ex)).

With matplotlib enabled, produces comparison plots for all quantities.

## Build and run

```bash
make run
```

## Output

- `exports/fmt_free_energy.png` — excess free energy per particle
- `exports/fmt_pressure.png` — compressibility factor
- `exports/fmt_chemical_potential.png` — excess chemical potential
