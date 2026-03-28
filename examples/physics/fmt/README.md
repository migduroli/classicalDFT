# Fundamental measure theory

## Overview

Demonstrates the FMT functional hierarchy (`dft_core::physics::fmt`): bulk
thermodynamics for all four models, the `Species` class for grid-based free
energy and force calculations, and the bounded alias mapping.

| Feature | What this example shows |
|---------|------------------------|
| Bulk free energy | $f_{\text{ex}}(\eta)$ for Rosenfeld, RSLT, White Bear I/II vs PY and CS |
| Chemical potential | $\mu_{\text{ex}}(\eta)$ comparison across models |
| Pressure | Compressibility factor $P/(\rho kT)$ from thermodynamic identity |
| f-functions | $f_1, f_2, f_3$ that define each FMT variant |
| FMT species | `compute_free_energy()` and `compute_forces()` on a 3D grid |
| Bounded alias | $\rho(y) = \rho_{\min} + c\,y^2/(1+y^2)$ mapping and its derivative |

## Running

```bash
make run-local
```

## Plots (requires Grace)

When built with `DFT_HAS_GRACE`, five plots are saved to `exports/`:

| File | Content |
|------|---------|
| `fmt_free_energy.png` | Excess free energy per particle for all FMT models vs PY/CS |
| `fmt_pressure.png` | Compressibility factor: FMT models vs exact EOS |
| `fmt_chemical_potential.png` | Excess chemical potential for Rosenfeld, WBI, WBII |
| `fmt_f_functions.png` | Model-defining $f_1, f_2, f_3$ functions of $\eta$ |
| `fmt_alias.png` | Bounded alias $\rho(y)$, $\eta(y)$, and $d\rho/dy$ |
