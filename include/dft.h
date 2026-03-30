#ifndef DFT_H
#define DFT_H

// Plotting
#ifdef DFT_HAS_GRACE
#include "dft/plotting/grace.h"
#endif

// Math
#include "dft/math/arithmetic.h"
#include "dft/math/fourier.h"
#include "dft/math/hessian.h"
#include "dft/math/integration.h"
#include "dft/math/spline.h"

// Config & Console
#include "dft/config.h"
#include "dft/console.h"

// Crystal
#include "dft/crystal/lattice.h"

// Density & Species
#include "dft/density.h"
#include "dft/species.h"

// Potentials
#include "dft/potentials/potential.h"

// Thermodynamics
#include "dft/thermodynamics/enskog.h"
#include "dft/thermodynamics/eos.h"

// Functional
#include "dft/functional/fmt/convolution.h"
#include "dft/functional/fmt/functional.h"
#include "dft/functional/fmt/measures.h"
#include "dft/functional/fmt/species.h"
#include "dft/functional/fmt/weights.h"
#include "dft/functional/interaction.h"

// Solver
#include "dft/solver.h"

// Dynamics
#include "dft/dynamics/fire2.h"
#include "dft/dynamics/integrator.h"
#include "dft/dynamics/minimizer.h"

// Geometry
#include "dft/geometry/2D/element.h"
#include "dft/geometry/2D/mesh.h"
#include "dft/geometry/2D/uniform_mesh.h"
#include "dft/geometry/3D/element.h"
#include "dft/geometry/3D/mesh.h"
#include "dft/geometry/3D/uniform_mesh.h"
#include "dft/geometry/base/element.h"
#include "dft/geometry/base/mesh.h"
#include "dft/geometry/base/vertex.h"

#endif  // DFT_H
