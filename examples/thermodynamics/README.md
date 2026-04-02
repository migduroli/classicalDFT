# Thermodynamics example

Demonstrates hard-sphere equations of state (Carnahan-Starling,
Percus-Yevick virial/compressibility), thermodynamic consistency checks,
Enskog transport coefficients, and full EOS models (ideal gas, PY, LJ-JZG,
LJ-Mecke).

With matplotlib enabled, produces pressure and viscosity plots.

## Build and run

```bash
make run
```

## Output

- `exports/hs_pressure.png` — hard-sphere compressibility factor
- `exports/transport_viscosity.png` — Enskog viscosities
