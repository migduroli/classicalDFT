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

---

## Step-by-step code walkthrough

### Step 1: QAGS — definite integral

An `Integrator` is constructed with a callable and evaluated:

```cpp
auto neg_exp = math::Integrator([](double x) { return std::exp(-x); });
auto r1 = neg_exp.integrate(0.0, -std::log(0.5));
```

This computes $\int_0^{-\ln 0.5} e^{-x}\,dx = 0.5$ using adaptive
Gauss-Kronrod subdivision.

### Step 2: QNG — fast non-adaptive

The same integrand using the fixed-order non-adaptive rule:

```cpp
auto r2 = neg_exp.integrate_fast(0.0, -std::log(0.5));
```

### Step 3: QAGIU — upper semi-infinite

$$
\int_0^\infty e^{-x}\,dx = 1
$$

```cpp
auto r3 = neg_exp.integrate_upper_infinite(0.0);
```

### Step 4: QAGIL — lower semi-infinite

$$
\int_{-\infty}^0 e^{x}\,dx = 1
$$

```cpp
auto pos_exp = math::Integrator([](double x) { return std::exp(x); });
auto r4 = pos_exp.integrate_lower_infinite(0.0);
```

### Step 5: QAGI — full infinite (standard normal)

$$
\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}\,dx = 1
$$

```cpp
auto gaussian = math::Integrator([](double x) {
    return std::exp(-x * x * 0.5) / std::sqrt(2.0 * std::numbers::pi);
});
auto r5 = gaussian.integrate_infinite();
```

Each result object has `.value` and `.error` (GSL-estimated absolute error).

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
