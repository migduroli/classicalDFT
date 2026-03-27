# Spline interpolation

## Overview

The `dft_core::numerics::spline` namespace wraps GSL's spline routines behind
a modern C++20 interface. All classes are move-only RAII types that accept
`std::span` inputs.

| Class | Purpose |
|-------|---------|
| `CubicSpline` | 1D natural cubic spline with evaluation, derivatives, and integration |
| `BivariateSpline` | 2D bicubic spline on a regular grid with all partial derivatives |

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::numerics::spline;

// 1D: interpolate sin(x)
auto x = std::vector<double>{0, 1, 2, 3, 4, 5, 6};
auto y = std::vector<double>{0, 0.84, 0.91, 0.14, -0.76, -0.96, -0.28};
auto s = CubicSpline(x, y);

double val    = s(2.5);         // evaluate
double dval   = s.derivative(2.5);   // first derivative
double d2val  = s.derivative2(2.5);  // second derivative
double integ  = s.integrate(0, M_PI); // definite integral

// 2D: interpolate f(x,y) = sin(x) * cos(y)
auto bx = std::vector<double>{ ... };
auto by = std::vector<double>{ ... };
auto bz = std::vector<double>{ ... };  // row-major: z[j + ny*i]
auto surface = BivariateSpline(bx, by, bz);

double val2  = surface(1.0, 2.0);
double dx    = surface.deriv_x(1.0, 2.0);
double dy    = surface.deriv_y(1.0, 2.0);
double dxy   = surface.deriv_xy(1.0, 2.0);
```

## Run

```bash
make run
```
