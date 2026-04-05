# Arithmetic: compensated summation

## Purpose

This example compares compensated summation algorithms against naive
`std::accumulate` on arrays designed to expose floating-point rounding errors.
These algorithms are used internally by the DFT library whenever sums over
large grids ($N \sim 10^5$--$10^6$ points) must be computed accurately, for
instance in total free energy evaluations and normalisation integrals.

## Mathematical background

### The problem: catastrophic cancellation

IEEE 754 double-precision arithmetic carries about 15--16 significant decimal
digits. When summing $N$ numbers, the naive left-to-right accumulation

$$
S = \sum_{i=1}^{N} x_i, \qquad \hat{S} = \mathrm{fl}\!\left(\cdots\mathrm{fl}(\mathrm{fl}(x_1 + x_2) + x_3) + \cdots + x_N\right)
$$

incurs a rounding error bounded by:

$$
|\hat{S} - S| \leq (N-1)\,\varepsilon_{\mathrm{mach}}\,\sum_{i=1}^{N}|x_i| + O(\varepsilon_{\mathrm{mach}}^2)
$$

where $\varepsilon_{\mathrm{mach}} \approx 1.1 \times 10^{-16}$ for `double`.
This error grows linearly with $N$ and can become significant for large sums,
especially when terms of opposite sign nearly cancel.

### Kahan-Babuska summation

Kahan's algorithm maintains a running compensation term $c$ that tracks the
accumulated rounding error:

$$
\begin{aligned}
y &= x_i - c \\
t &= S + y \\
c &= (t - S) - y \\
S &= t
\end{aligned}
$$

The error bound improves to $O(\varepsilon_{\mathrm{mach}})$ regardless of $N$:

$$
|\hat{S} - S| \leq 2\varepsilon_{\mathrm{mach}}\,|S| + O(\varepsilon_{\mathrm{mach}}^2)
$$

### Neumaier summation

Neumaier's variant of Kahan summation handles the case where $|x_i| > |S|$
(i.e. the new term is larger than the accumulated sum). It checks which
operand has the larger magnitude and compensates accordingly:

```
if |S| >= |x_i|:
    c += (S - t) + x_i
else:
    c += (x_i - t) + S
```

This makes it numerically stable even when terms arrive in decreasing order.

### Klein summation (second-order)

Klein's algorithm applies the Kahan compensation recursively to the error
term itself, achieving second-order compensation. The error bound is
$O(\varepsilon_{\mathrm{mach}}^2)$ independent of $N$. This is the most
accurate variant but involves more operations per element.

---

## Step-by-step code walkthrough

Two test arrays are constructed with values that differ at the level of
machine epsilon:

```cpp
std::vector<double> x1 = {1.0 + 1e-14, 2.5 + 1e-14, 3.0 + 1e-14, 4.0 + 1e-14};
std::vector<double> x2 = {1.00100001, 2.50010002, 3.00020001, 4.00010003};
```

Each array is summed four ways:

1. **`std::accumulate`** — naive left-to-right accumulation.
2. **`math::kahan_sum`** — Kahan-Babuska compensated sum.
3. **`math::neumaier_sum`** — Neumaier compensated sum.
4. **`math::klein_sum`** — Klein second-order compensated sum.

Output is printed at full `double` precision (18 significant digits) so that
the last few digits reveal the rounding differences between algorithms.

For these small test arrays the differences are tiny (at the
$10^{-16}$--$10^{-15}$ level), but the effect becomes pronounced when summing
$10^5$+ terms, as occurs in the DFT grid integrals.

## Build and run

```bash
make run-local
```
