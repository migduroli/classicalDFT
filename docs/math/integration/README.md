# Integration: GSL quadrature

## Purpose

This example demonstrates the GSL-based `Integrator` class with all supported
quadrature methods. These numerical integration routines are used throughout
the DFT library for computing the Barker-Henderson hard-sphere diameter, the
van der Waals attractive parameter, equation-of-state integrals, and other
one-dimensional integrals that arise in the bulk thermodynamics.

## Mathematical background

### Adaptive Gauss-Kronrod quadrature (QAGS)

The QAGS algorithm approximates a definite integral:

$$
I = \int_a^b f(x)\,dx
$$

using an adaptive subdivision strategy. Each subinterval is evaluated with a
Gauss-Kronrod rule pair: a $G$-point Gauss rule and a $K$-point Kronrod rule
($K = 2G + 1$). The difference between the two estimates provides an error
estimate for that subinterval. QAGS refines the subinterval with the largest
estimated error until the global error satisfies:

$$
|I_{\mathrm{approx}} - I| \leq \max(\varepsilon_{\mathrm{abs}},\; \varepsilon_{\mathrm{rel}}\,|I|)
$$

The default tolerances are $\varepsilon_{\mathrm{abs}} = 10^{-12}$ and
$\varepsilon_{\mathrm{rel}} = 10^{-8}$, with a workspace of 1000 subintervals.

### Non-adaptive Gauss-Kronrod (QNG)

The QNG method applies a fixed-order Gauss-Kronrod rule (10-point, 21-point,
43-point, or 87-point) without subdivision. It is faster than QAGS but only
suitable for smooth integrands. If the requested accuracy cannot be achieved,
it reports the estimated error without attempting refinement.

### Semi-infinite and infinite intervals

GSL transforms unbounded integrals to bounded ones via variable substitution:

- **QAGIU** $[a, \infty)$: the substitution $x = a + (1-t)/t$ maps the
  integral to $\int_0^1$:
  $$
  \int_a^\infty f(x)\,dx = \int_0^1 f\!\left(a + \frac{1-t}{t}\right)\frac{1}{t^2}\,dt
  $$

- **QAGIL** $(-\infty, b]$: the substitution $x = b - (1-t)/t$ maps the
  integral to $\int_0^1$:
  $$
  \int_{-\infty}^b f(x)\,dx = \int_0^1 f\!\left(b - \frac{1-t}{t}\right)\frac{1}{t^2}\,dt
  $$

- **QAGI** $(-\infty, \infty)$: splits at 0 and applies QAGIU to $[0,\infty)$
  and QAGIL to $(-\infty, 0]$.

The transformed integrand is then evaluated using the adaptive QAGS algorithm.

## What the code does

### 1. QAGS: definite integral

$$
\int_0^{-\ln 0.5} e^{-x}\,dx = 1 - e^{\ln 0.5} = 0.5
$$

### 2. QNG: same integral, non-adaptive

Same integrand and limits, using `integrate_fast()`. The result should agree
with QAGS; the reported error may be slightly larger due to the fixed-order
rule.

### 3. QAGIU: upper semi-infinite

$$
\int_0^\infty e^{-x}\,dx = 1
$$

### 4. QAGIL: lower semi-infinite

$$
\int_{-\infty}^0 e^{x}\,dx = 1
$$

### 5. QAGI: full infinite (standard normal)

$$
\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}\,dx = 1
$$

Each result is printed with the computed value and the GSL-estimated absolute
error.

## API design

The `math::Integrator` is constructed with a callable (lambda, function
pointer, or `std::function`):

```cpp
auto integrator = math::Integrator([](double x) { return std::exp(-x); });
auto result = integrator.integrate(a, b);  // result.value, result.error
```

The `Integrator` manages a GSL workspace internally (RAII). It provides five
methods corresponding to the five quadrature schemes described above.

## Build and run

```bash
make run-local
```
