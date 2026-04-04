# Solver: phase diagram computation

## Physical background

A fluid with attractive interactions undergoes a first-order liquid-vapor
phase transition below a critical temperature $T_c$. The phase diagram is
the collection of coexistence (binodal) and mechanical instability (spinodal)
curves in the $(\rho, T)$ plane.

### Pressure and the van der Waals loop

The pressure of the mean-field DFT fluid is:

$$
P(\rho, T) = \rho k_BT\left[1 + \rho\,\phi_{\mathrm{ex}}'(\rho)\right]
$$

where $\phi_{\mathrm{ex}}(\rho) = f_{\mathrm{HS}}(\eta) + \tfrac{1}{2}a_{\mathrm{vdw}}\rho$
is the total excess free energy per particle (hard-sphere plus mean-field attraction).

Below $T_c$, the $P(\rho)$ isotherm displays a van der Waals loop with a
local maximum and minimum, signalling phase coexistence. Between the two
extrema lies the mechanically unstable region where $\partial P/\partial\rho < 0$.

### Spinodal

The spinodal curve marks the boundary of mechanical stability:

$$
\left.\frac{\partial P}{\partial \rho}\right|_{T} = 0
$$

The library finds the spinodal densities by scanning $\rho$ for sign changes
in $\partial P/\partial\rho$ and refining by bisection.

### Coexistence (binodal)

The coexistence (binodal) curve satisfies the equal-pressure and
equal-chemical-potential conditions simultaneously:

$$
P(\rho_v, T) = P(\rho_l, T), \qquad \mu(\rho_v, T) = \mu(\rho_l, T)
$$

where $\rho_v$ and $\rho_l$ are the vapor and liquid coexistence densities.
This is equivalent to the Maxwell equal-area construction on the van der
Waals loop. The library solves this system via Newton iteration with
bisection fallback.

### Pseudo-arclength continuation

To trace the full binodal curve including the critical point, the library
uses pseudo-arclength continuation. Starting from a low-temperature
coexistence point, the curve is parameterised by arclength $s$:

$$
\frac{d}{ds}\begin{pmatrix}\rho_v \\ \rho_l \\ T\end{pmatrix}
$$

with the tangent predictor and Newton corrector. This naturally handles the
turning point (pitchfork bifurcation) at $T_c$ where the binodal curve folds.

### Critical point

At the critical point, the spinodal and binodal curves meet. The critical
temperature $T_c$ and density $\rho_c$ satisfy:

$$
\frac{\partial P}{\partial\rho} = 0, \qquad \frac{\partial^2 P}{\partial\rho^2} = 0
$$

The continuation algorithm automatically identifies $T_c$ as the point where
$\rho_v \to \rho_l$.

## What the code does

1. Evaluates $P^*(\rho)$ at 6 temperatures ($T^* = 0.6$ to $1.2$) showing
   the van der Waals loop below $T_c$.
2. Traces the binodal curve for all four FMT models (Rosenfeld, RSLT,
   White Bear I, White Bear II) using pseudo-arclength continuation.
3. Traces the spinodal curve for each model.
4. Demonstrates spline interpolation of the phase boundaries at arbitrary
   temperatures.

## Cross-validation (`check/`)

| Step | Category | Method (ours) | Method (Jim's) | Grid | Tolerance |
|------|----------|--------------|----------------|------|-----------|
| 1-3 | Spinodal $\rho_{\mathrm{low}}, \rho_{\mathrm{high}}$ | Bisection on $\partial P/\partial\rho$ sign | Golden section on $P(\rho)$ extrema | $kT = 0.7, 0.8, 0.9$ | $10^{-6}$ |
| 4-6 | Coexistence $\rho_v, \rho_l$ | Newton on equal-$\mu$ | Bisection on $\Delta P$ | $kT = 0.7, 0.8, 0.9$ | $10^{-6}$ |
| 7 | Binodal curve | Pseudo-arclength continuation | Point-by-point findCoex | 3 interior temperatures | $10^{-4}$ |
| 7 | Critical point $T_c$ | Continuation endpoint | N/A | single | $T_c \in [1.0, 1.5]$ |
| 7 | Density gap at $T_c$ | $|\rho_v - \rho_l|$ at end | N/A | single | $< 0.05$ |

Both methods verify thermodynamic consistency: $|P_v - P_l| < 10^{-10}$ and
$|\mu_v - \mu_l| < 10^{-10}$ at every coexistence point.

## Build and run

```bash
make run        # Docker
make run-local  # local build
make run-checks # cross-validation against Jim's code
```

## Output

### Pressure isotherms

Six isotherms from $T^* = 0.6$ (deep sub-critical, large van der Waals loop)
to $T^* = 1.2$ (above critical, monotonic).

![Isotherms](exports/isotherms.png)

### Phase diagram (all FMT models)

Binodal (solid) and spinodal (dashed, transparent) curves for all four FMT
models, with critical points marked.

![Coexistence](exports/coexistence.png)

### Phase diagram (White Bear II)

Binodal and spinodal for White Bear II alone, showing the two-phase region
bounded by the coexistence dome and the mechanical instability region bounded
by the spinodal.

![Phase diagram](exports/phase_diagram.png)
