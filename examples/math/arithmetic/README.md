# Arithmetic: compensated summation

Compares compensated summation algorithms against naive `std::accumulate` on
arrays designed to expose floating-point rounding errors.

## What this example does

Sums two test arrays (with small perturbations near machine epsilon) using
four algorithms: naive accumulation, Kahan-Babuska, Neumaier, and Klein.
Output is printed at full `double` precision (18 digits) to reveal the
rounding differences.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `math::kahan_sum()` | Kahan-Babuska compensated sum |
| `math::neumaier_sum()` | Neumaier compensated sum |
| `math::klein_sum()` | Klein second-order compensated sum |

## Build and run

```bash
make run
```
