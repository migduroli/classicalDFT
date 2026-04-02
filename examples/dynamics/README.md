# Dynamics example

Demonstrates two algorithm modules:

1. **FIRE2 minimizer** (`algorithms::fire`): minimises a 2D quadratic using both
   the one-shot `minimize()` API and the step-by-step `initialize()` + `step()`
   loop for energy logging.

2. **Split-operator DDFT** (`algorithms::ddft`): evolves a sinusoidal density
   perturbation in an ideal gas, showing variance decay and mass conservation.

With matplotlib enabled, produces energy convergence and variance decay plots.

## Build and run

```bash
make run
```

## Output

- `exports/fire2_energy.png` — FIRE2 energy convergence
- `exports/ddft_variance.png` — DDFT density variance decay
