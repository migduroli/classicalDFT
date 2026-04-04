# Autodiff: automatic differentiation

## Purpose

This doc demonstrates the automatic differentiation (AD) adapter that
provides exact derivatives of scalar functions without manual derivation or
finite-difference approximation. The library uses AD internally to compute
derivatives of the FMT free energy density $\Phi(\{n_\alpha\})$ with respect
to the weighted densities, and derivatives of equations of state with respect
to density.

## Mathematical background

### Forward-mode automatic differentiation

Forward-mode AD propagates derivatives alongside function values by
replacing each real number $x$ with a dual number $x + \epsilon\,\dot{x}$,
where $\epsilon^2 = 0$. Applying a function $f$ to this dual number yields:

$$
f(x + \epsilon\,\dot{x}) = f(x) + \epsilon\,f'(x)\,\dot{x}
$$

so $f'(x)$ is extracted from the $\epsilon$-coefficient. Higher-order
derivatives use hyper-dual numbers: `dual2nd` carries $(f, f', f'')$ and
`dual3rd` carries $(f, f', f'', f''')$.

### Comparison with finite differences

The central finite difference approximation:

$$
f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}
$$

introduces a truncation error of $O(h^2)$ and a round-off error of
$O(\varepsilon_{\mathrm{mach}}/h)$. The optimal step size
$h \sim \varepsilon_{\mathrm{mach}}^{1/3} \approx 10^{-5}$ gives at best
$\sim 10^{-10}$ accuracy.

Autodiff computes derivatives to machine precision ($\sim 10^{-16}$) with
no step size tuning, and no cancellation error.

### API

| Function | Returns | Dual type |
|----------|---------|-----------|
| `derivatives_up_to_1(f, x)` | $(f, f')$ | `dual` |
| `derivatives_up_to_2(f, x)` | $(f, f', f'')$ | `dual2nd` |
| `derivatives_up_to_3(f, x)` | $(f, f', f'', f''')$ | `dual3rd` |

The function `f` must be written in terms of autodiff-compatible operations
(the standard math functions are overloaded in the `autodiff::detail`
namespace).

## What the code does

1. **First derivatives**: computes $f'(x)$ for $\sin(x)$, $\exp(x)$, and
   a cubic polynomial, comparing against exact analytical values.

2. **Second derivatives**: computes $f''(x)$ for $\sin(x)$ and $\exp(x)$,
   verifying that $\sin''(x) = -\sin(x)$ and $\exp''(x) = \exp(x)$.

3. **Third derivatives**: computes $f'''(x)$ for $\sin(x)$ and the cubic
   polynomial (whose third derivative is the constant $6$).

4. **Autodiff vs finite differences**: for $f(x) = \ln(1 + x^2)$, compares
   the accuracy of autodiff against central finite differences with
   $h = 10^{-5}$. Autodiff achieves $\sim 10^{-16}$ error while finite
   differences reach only $\sim 10^{-10}$.

## Build and run

```bash
make run-local
```
