#pragma once

// ── Core ────────────────────────────────────────────────────────────────────
#include "cdft/core/types.hpp"
#include "cdft/core/grid.hpp"
#include "cdft/config.hpp"

// ── Numerics ────────────────────────────────────────────────────────────────
#include "cdft/numerics/autodiff.hpp"
#include "cdft/numerics/math.hpp"
#include "cdft/numerics/fourier.hpp"
#include "cdft/numerics/spline.hpp"

// ── Physics ─────────────────────────────────────────────────────────────────
#include "cdft/physics/potentials.hpp"
#include "cdft/physics/eos.hpp"
#include "cdft/physics/crystal.hpp"

// ── Functional ──────────────────────────────────────────────────────────────
#include "cdft/functional/fmt.hpp"
#include "cdft/functional/density.hpp"
#include "cdft/functional/species.hpp"
#include "cdft/functional/interaction.hpp"

// ── Visualization ───────────────────────────────────────────────────────────
#include "cdft/viz/plot.hpp"
