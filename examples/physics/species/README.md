# Species

## Overview

Demonstrates the `dft_core::physics::species::Species` class: alias coordinates,
external fields, the fixed-mass constraint, and the force protocol.

| Feature | What this example shows |
|---------|------------------------|
| Alias coordinates | $\rho = \rho_{\min} + x^2$ mapping, round-trip fidelity |
| External field | Gravitational slab: $V_{\text{ext}} = gz$, barometric density $\rho \propto e^{-gz}$ |
| Fixed-mass constraint | Rescaling and Lagrange multiplier projection |
| Force protocol | `begin_force_calculation()` / accumulate / `end_force_calculation()` |
| Alias chain rule | $\partial F / \partial x = 2x \, \partial F / \partial \rho$ |

## Running

```bash
make run-local
```

## Plots (requires Grace)

When built with `DFT_HAS_GRACE`, three plots are saved to `exports/`:

| File | Content |
|------|---------|
| `barometric_density.png` | Barometric density $\rho(z) \propto e^{-gz}$ and linear external field |
| `alias_mapping.png` | Alias coordinate $x(\rho)$ and sensitivity $dx/d\rho = 1/(2x)$ |
| `force_profiles.png` | Force in $\rho$-space and $x$-space along $z$ |
