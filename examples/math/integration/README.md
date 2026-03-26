# Numerical integration (GSL wrapper)

## Overview

The `dft::numerics::integration` namespace wraps several GSL quadrature
routines behind a clean C++ interface. Two wrapper classes are provided:

| Class | Use case |
|-------|----------|
| `Integrator<T>` | Integrate a member function of an object of type `T` |
| `FunctionIntegrator<P>` | Integrate a free function with parameter pack `P` |

Both support five quadrature methods:

| Method | GSL routine | Domain |
|--------|-------------|--------|
| `definite_integral` | `gsl_integration_qags` | [a, b] |
| `definite_integral_fast` | `gsl_integration_qng` | [a, b] (non-adaptive) |
| `upper_semi_infinite_integral` | `gsl_integration_qagiu` | [a, +inf) |
| `lower_semi_infinite_integral` | `gsl_integration_qagil` | (-inf, b] |
| `full_infinite_integral` | `gsl_integration_qagi` | (-inf, +inf) |

## Usage

```cpp
#include <classicaldft>
using namespace dft::numerics::integration;

// Define a problem class with a member function to integrate
struct Problem {
  double f(double x) const { return exp(-x); }
};

Problem p;
auto integrator = Integrator<Problem>(p, &Problem::f);

// Finite interval: int_0^1 e^{-x} dx = 1 - e^{-1}
double result = integrator.definite_integral(0, 1);

// Semi-infinite: int_0^{+inf} e^{-x} dx = 1
double result_inf = integrator.upper_semi_infinite_integral(0);

// Free function with parameters
auto params = std::vector<double>{2.0};
auto func_int = FunctionIntegrator<std::vector<double>>(neg_exp, params);
double r = func_int.definite_integral(0, 1);
```

## Running

```bash
make run   # builds and runs inside Docker
```
