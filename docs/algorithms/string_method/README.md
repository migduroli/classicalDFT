# String method: minimum free-energy path

## Purpose

This doc demonstrates the string method for computing minimum
free-energy paths (MEPs) between two metastable states. The library
provides two overloads, corresponding to the two variants in the
literature:

- **Full string method** (4-argument `find_pathway`): evolves images
  using only the perpendicular component of the force, guaranteeing
  convergence to zero perpendicular force by construction.
- **Simplified string method** (5-argument `find_pathway`): relaxes
  images with a user-supplied full-gradient method (DDFT, FIRE, etc.)
  and relies on reparametrization to remove tangential drift.

Both operate on discretized fields with user-supplied energy functions.
No DFT-specific code is required.

## Mathematical background

### Minimum free-energy path

Given a free-energy functional $F[\mathbf{x}]$, the MEP between states
$\mathbf{x}_A$ and $\mathbf{x}_B$ is the curve $\varphi(s)$,
$s \in [0, 1]$, satisfying:

$$
(\nabla F)\!\!\perp = 0
$$

where $\perp$ denotes the component perpendicular to the path tangent.
Along the MEP, the force is purely tangential.

### Full string method

The full string method (E, Ren, Vanden-Eijnden, Phys. Rev. B **66**,
052301, 2002) discretizes $\varphi$ into $M$ images
$\{\mathbf{x}_0, \ldots, \mathbf{x}_{M-1}\}$ and iterates:

1. **Perpendicular evolution**: compute the gradient $\nabla F$ at each
   interior image, project out the tangential component using the
   finite-difference tangent from neighbours, and evolve
   $\mathbf{x}_j \leftarrow \mathbf{x}_j - \alpha\,(\nabla F)_\perp$
   for a fixed number of steps.
2. **Reparametrization**: redistribute images at equal arc-length spacing.

Because only the perpendicular force drives the evolution, this variant
converges the path to zero perpendicular force (the MEP condition) by
construction. Convergence is measured as $\max_j |(\nabla F)_\perp|_j$.

Usage (4-argument overload, no relaxation callback):

```cpp
algorithms::string_method::StringMethod sm{
    .tolerance = 1e-6,
    .max_iterations = 200,
    .evolution_steps = 20,
    .evolution_alpha = 0.01,
};

auto result = sm.find_pathway(state_a, state_b, num_images, energy_fn);
```

### Simplified string method

The simplified string method (E, Ren, Vanden-Eijnden, J. Chem. Phys.
**126**, 164103, 2007) uses the same image discretization but replaces
the perpendicular evolution with a user-supplied relaxation:

1. **Relaxation**: evolve each interior image under the full gradient
   $\nabla F$ (steepest descent, DDFT, FIRE, or any other method).
2. **Reparametrization**: redistribute images at equal arc-length spacing
   to prevent clustering near energy minima.

Step 2 removes the tangential drift without requiring explicit tangent
estimates. Convergence is measured as the RMS change in image energies
between iterations.

Usage (5-argument overload, with relaxation callback):

```cpp
algorithms::string_method::StringMethod sm{
    .tolerance = 1e-8,
    .max_iterations = 200,
};

auto result = sm.find_pathway(state_a, state_b, num_images, energy_fn, relax_fn);
```

This variant is useful when a physics-based relaxation (e.g. DDFT) is
available. The user controls the relaxation dynamics; the string method
only handles reparametrization and convergence checking.

### Arc-length reparametrization

After each evolution/relaxation step, the cumulative arc length is:

$$
\alpha_j = \sum_{k=1}^{j} \|\mathbf{x}_k - \mathbf{x}_{k-1}\|
$$

Images are redistributed so that $\alpha_j = j \cdot L / (M - 1)$
where $L = \alpha_{M-1}$ is the total path length. Linear interpolation
within each bracketing interval gives the new positions.

## Example

The example uses a two-component potential with a curved valley:

$$
V(x, y) = (x^2 - 1)^2 + 10\,(y - x^2)^2
$$

The two minima sit at $(-1, 1)$ and $(+1, 1)$, both with $V = 0$.
The valley follows the parabola $y = x^2$.

The linear interpolation between the endpoints stays at $y = 1$,
passing over a high barrier $V(0, 1) = 11$. The true MEP follows
the parabolic valley with a saddle-point barrier of only $V(0, 0) = 1$.
The string method reduces the barrier from 11 to 1 and recovers the
curved path in about 6 iterations.

Note that the true MEP does not exactly follow the valley floor
$y = x^2$.  On the valley floor, $\partial V / \partial y = 0$, but
the full perpendicular force $|\nabla V_\perp|$ is nonzero because the
path tangent has a $y$-component due to the curvature of $y = x^2$.
Analytically, $|F_\perp|_{y=x^2} = 8x^2|x^2-1|/\sqrt{1+4x^2}$,
which reaches ~1.16 at $x \approx 0.7$.  The converged MEP deviates
from $y = x^2$ by up to ~0.04, sitting inside the curve, and achieves
$\max|F_\perp| < 10^{-6}$.

This problem also demonstrates the multi-component interface: each
image state consists of two `arma::vec` entries (one for $x$, one
for $y$).

## Results

### Energy landscape and MEP

The contour plot shows $\log_{10} V(x, y)$ with the initial straight-line
path (dashed brown) and the converged MEP (solid orange). The converged
path follows the parabolic valley $y = x^2$ through the saddle at the
origin, avoiding the high-energy ridge at $y = 1$.

![Energy landscape and MEP](exports/string_method_landscape.pdf)

### Energy along the path

The energy profile along the arc-length parameter shows the barrier
reduction from $V \approx 11$ (initial linear path) to $V = 1$
(converged MEP through the saddle).

![Energy along the path](exports/string_method_energy.pdf)

### Convergence

The max perpendicular force across images decreases rapidly, reaching the
tolerance of $10^{-6}$ within a few tens of iterations.

![Convergence](exports/string_method_convergence.pdf)

## References

- E, W., Ren, W., Vanden-Eijnden, E. "String method for the study of
  rare events", Phys. Rev. B **66**, 052301 (2002).
- E, W., Ren, W., Vanden-Eijnden, E. "Simplified and improved string
  method for computing the minimum energy path", J. Chem. Phys. **126**,
  164103 (2007).
- Lutsko, J. F. "How crystals form: a theory of nucleation pathways",
  Sci. Adv. **5**, eaav7399 (2019).
