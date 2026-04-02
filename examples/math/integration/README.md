# Integration: GSL quadrature

Demonstrates the GSL-based `Integrator` class with all supported quadrature
methods.

## What this example does

1. **QAGS** (adaptive): $\int_0^{-\ln 0.5} e^{-x} \, dx = 0.5$.
2. **QNG** (non-adaptive, fast): same integral for comparison.
3. **QAGIU** (upper semi-infinite): $\int_0^\infty e^{-x} \, dx = 1$.
4. **QAGIL** (lower semi-infinite): $\int_{-\infty}^0 e^{x} \, dx = 1$.
5. **QAGI** (full infinite): $\int_{-\infty}^\infty \mathcal{N}(x) \, dx = 1$
   where $\mathcal{N}(x)$ is the standard normal distribution.

Each result prints the value and estimated absolute error.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `math::Integrator` | RAII quadrature wrapper |
| `integrate()` | finite interval (QAGS) |
| `integrate_fast()` | non-adaptive (QNG) |
| `integrate_upper_infinite()` | $[a, \infty)$ (QAGIU) |
| `integrate_lower_infinite()` | $(-\infty, b]$ (QAGIL) |
| `integrate_infinite()` | $(-\infty, \infty)$ (QAGI) |

## Build and run

```bash
make run
```
