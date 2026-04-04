# Interaction: mean-field attraction

## Overview

In thermodynamic perturbation theory the attractive tail of the pair potential
is treated at the mean-field level. This example evaluates the mean-field
free energy, compares WCA and BH splitting schemes, and studies the
convergence of the discrete grid weights toward the continuum integral.

## The mean-field free energy functional

$$
F_{\mathrm{mf}}[\rho] = \frac{1}{2}\int\!\int \rho(\mathbf{r})\, w_{\mathrm{att}}(|\mathbf{r}-\mathbf{r}'|)\, \rho(\mathbf{r}')\, d\mathbf{r}\, d\mathbf{r}'
$$

where $w_{\mathrm{att}}(r) = v_{\mathrm{att}}(r)/(k_BT)$ is the attractive
tail of the pair potential divided by temperature.

## Perturbation splitting schemes

The decomposition $v(r) = v_{\mathrm{rep}}(r) + v_{\mathrm{att}}(r)$ is not unique.
Two standard choices are implemented:

### Weeks-Chandler-Andersen (WCA)

Splits at the potential minimum $r_{\min} = 2^{1/6}\sigma$:

$$
w_{\mathrm{att}}^{\mathrm{WCA}}(r) = \begin{cases}
v(r_{\min})/kT, & r < r_{\min} \\
v(r)/kT, & r_{\min} \leq r < r_c \\
0, & r \geq r_c
\end{cases}
$$

WCA is the natural choice for perturbation DFT because the reference system
(purely repulsive WCA potential) is well approximated by hard spheres.

### Barker-Henderson (BH)

Splits at the zero crossing $r_0 = \sigma$ where $v(r_0) = 0$:

$$
w_{\mathrm{att}}^{\mathrm{BH}}(r) = \begin{cases}
0, & r < r_0 \\
v(r)/kT, & r_0 \leq r < r_c \\
0, & r \geq r_c
\end{cases}
$$

## Van der Waals parameter

The spatially integrated weight defines the mean-field coupling constant:

$$
a_{\mathrm{vdw}} = \int_0^{r_c} w_{\mathrm{att}}(r)\, 4\pi r^2\, dr
$$

For the Lennard-Jones potential with WCA splitting at cutoff $r_c$, the
analytical integral is:

$$
a_{\mathrm{vdw}}^{\mathrm{WCA}} = \frac{4\pi}{kT}\left[\frac{4\varepsilon\sigma^{12}}{9}\left(r_{\min}^{-9} - r_c^{-9}\right) - \frac{4\varepsilon\sigma^6}{3}\left(r_{\min}^{-3} - r_c^{-3}\right) + \frac{v(r_{\min})}{3}\left(r_{\min}^3 - r_c^3\right) \right]
$$

where the first two terms come from integrating the LJ tail from $r_{\min}$
to $r_c$, and the third from integrating the constant $v(r_{\min})$ from $0$
to $r_{\min}$.

In the bulk (uniform density) limit the mean-field free energy simplifies to
$F_{\mathrm{mf}} = \tfrac{1}{2}a_{\mathrm{vdw}}\rho^2 V$, so all bulk
thermodynamics depend on $a_{\mathrm{vdw}}$ alone.

## Discrete weights and grid convergence

On a discrete grid with spacing $\Delta x$, the continuum convolution is
replaced by a discrete sum:

$$
F_{\mathrm{mf}} = \frac{1}{2}\sum_{i,j} \rho_i\, w_{ij}\, \rho_j\, (\Delta x)^6
$$

The weights $w_{ij}$ depend only on the displacement $(i-j)$ (translation
invariance). For each cell displacement $(i_x, i_y, i_z)$ the library
computes the weight by numerical quadrature.

### Quadrature schemes

The library supports three schemes of increasing accuracy:

**InterpolationZero**: point evaluation at the cell centre:

$$
w_{ijk} = w_{\mathrm{att}}\!\left(\sqrt{(i_x\Delta x)^2 + (i_y\Delta x)^2 + (i_z\Delta x)^2}\right)
$$

**InterpolationLinearF**: 8-point trilinear interpolation. Evaluates
$w_{\mathrm{att}}$ at the 8 corners of the cell and averages.

**InterpolationQuadraticF**: 27-point quadrature (Jim's QF scheme). For each
cell displacement, the weight function is evaluated at a $3\times3\times3$
sub-grid within the cell and combined with the quadrature coefficients. This
is the most accurate scheme and matches Jim's `Interaction.cpp` to $10^{-10}$.

### Grid convergence

The discrete $a_{\mathrm{vdw}} = \sum_{ijk} w_{ijk}\,(\Delta x)^3$ converges
toward the analytical value as $\Delta x \to 0$. At coarse grids
($\Delta x = 0.5\sigma$) the error can reach several percent; at
$\Delta x = 0.1\sigma$ it is below $10^{-4}$.

## What the code does

1. Evaluates $w_{\mathrm{att}}(r)$ under both WCA and BH splitting.
2. Computes $a_{\mathrm{vdw}}$ analytically.
3. Evaluates $f_{\mathrm{mf}}(\rho)$ and $\mu_{\mathrm{mf}}(\rho)$ from bulk weights.
4. Builds mean-field weights at 8 grid spacings and tracks the convergence of
   $a_{\mathrm{vdw}}$ toward the analytical value.

## Cross-validation (`check/`)

| Step | Category | Quantities | Grid | Tolerance |
|------|----------|-----------|------|-----------|
| 1 | Cell-by-cell QF weights | $w_{ijk}$ for all displacements within $r_c$ | $\Delta x = 0.4$ | $10^{-10}$ |
| 2 | Cell-by-cell QF weights | $w_{ijk}$ for all displacements within $r_c$ | $\Delta x = 0.2$ | $10^{-10}$ |
| 3 | Grid $a_{\mathrm{vdw}}$ | $\sum w_{ijk}\,(\Delta x)^3$ | $\Delta x = 0.5, 0.4, 0.3, 0.2, 0.1$ | $10^{-10}$ |
| 4 | Analytical $a_{\mathrm{vdw}}$ | Continuum integral (independent check) | N/A | reference |

All comparisons use Jim's `Interaction.cpp` (`generate_weight_QF`) as reference.

## Build and run

```bash
make run        # Docker
make run-local  # local build
make run-checks # cross-validation against Jim's code
```

## Output

### WCA vs BH splitting

![WCA vs BH](exports/interaction_wca_bh.png)

### Mean-field free energy density

![Free energy](exports/interaction_free_energy.png)

### Grid convergence of $a_{\mathrm{vdw}}$

![Grid convergence](exports/interaction_convergence.png)
