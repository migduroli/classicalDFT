# Density: full DFT pipeline

Demonstrates the complete classical DFT workflow:

1. Define a `physics::Model` with grid, species, interactions, and temperature
2. Create states via `init::homogeneous()` and `init::from_profile()`
3. Compute bulk thermodynamics: free energy, chemical potential, pressure
4. Find liquid-vapor coexistence
5. Evaluate the full DFT functional on homogeneous and inhomogeneous states
6. Construct a liquid slab profile and extract 1D cross-sections

## Build and run

```bash
make run-local
```

## Output

- `exports/pressure_isotherm.png` — Pressure isotherm with coexistence tie-line
- `exports/density_profile.png` — Liquid slab density profile
- `exports/free_energy.png` — Bulk free energy density
