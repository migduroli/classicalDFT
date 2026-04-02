# Spline: cubic and bivariate interpolation

Demonstrates the GSL-backed spline interpolation classes.

## What this example does

### CubicSpline (1D)

1. Interpolates $\sin(x)$ from 10 knots over $[0, 2\pi]$ and evaluates at 20
   points, printing the interpolation error at each.
2. Evaluates the first and second derivatives at $x = \pi/4$ and compares with
   the exact $\cos(\pi/4)$ and $-\sin(\pi/4)$.
3. Integrates the spline over $[0, \pi]$ and compares with the exact value 2.

### BivariateSpline (2D)

1. Interpolates $\sin(x) \cos(y)$ over $[0, 2\pi]^2$ from a $20 \times 20$
   grid and evaluates at four test points.
2. Evaluates partial derivatives $\partial f / \partial x$,
   $\partial f / \partial y$, and $\partial^2 f / \partial x \partial y$ at
   $(\pi/4, \pi/4)$ and compares with the exact values.

## Key API functions used

| Function | Purpose |
|----------|---------|
| `math::CubicSpline` | 1D cubic spline interpolation |
| `CubicSpline::derivative()` | first derivative |
| `CubicSpline::derivative2()` | second derivative |
| `CubicSpline::integrate()` | definite integral |
| `math::BivariateSpline` | 2D surface interpolation |
| `BivariateSpline::deriv_x()` / `deriv_y()` / `deriv_xy()` | partial derivatives |

## Build and run

```bash
make run
```
