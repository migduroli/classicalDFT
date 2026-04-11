#ifndef DFT_HPP
#define DFT_HPP

// Umbrella header for the classicalDFT library.
// Include this single header to access the full public API.

// Core types and grid.
#include "dft/console.hpp"
#include "dft/exceptions.hpp"
#include "dft/fields.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/types.hpp"

// Math utilities.
#include "dft/math/arithmetic.hpp"
#include "dft/math/fourier.hpp"
#include "dft/math/integration.hpp"
#include "dft/math/spline.hpp"

// Physics.
#include "dft/physics/eos.hpp"
#include "dft/physics/hard_spheres.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/physics/model.hpp"
#include "dft/physics/potentials.hpp"
#include "dft/physics/walls.hpp"

// Functionals.
#include "dft/functionals/bulk/coexistence.hpp"
#include "dft/functionals/bulk/phase_diagram.hpp"
#include "dft/functionals/bulk/thermodynamics.hpp"
#include "dft/functionals/external_field.hpp"
#include "dft/functionals/fmt/measures.hpp"
#include "dft/functionals/fmt/models.hpp"
#include "dft/functionals/functional.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/functionals/hard_sphere.hpp"
#include "dft/functionals/ideal_gas.hpp"
#include "dft/functionals/mean_field.hpp"
#include "dft/functionals/types.hpp"

// Algorithms.
#include "dft/algorithms/dynamics.hpp"
#include "dft/algorithms/fire.hpp"
#include "dft/algorithms/minimization.hpp"
#include "dft/algorithms/picard.hpp"
#include "dft/algorithms/saddle_point.hpp"
#include "dft/algorithms/solvers/continuation.hpp"
#include "dft/algorithms/solvers/gmres.hpp"
#include "dft/algorithms/solvers/jacobian.hpp"
#include "dft/algorithms/solvers/newton.hpp"

// Geometry.
#include "dft/geometry/element.hpp"
#include "dft/geometry/mesh.hpp"
#include "dft/geometry/vertex.hpp"

// Configuration.
#include "dft/config/parser.hpp"

// Plotting.
#include "dft/plotting/exceptions.hpp"
#include "dft/plotting/matplotlib.hpp"

#endif // DFT_HPP
