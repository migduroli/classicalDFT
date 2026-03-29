#pragma once

#include "cdft/functional/density.hpp"
#include "cdft/functional/species.hpp"
#include "cdft/physics/potentials.hpp"

#include <armadillo>

namespace cdft::functional {

  enum class WeightScheme {
    InterpolationZero,
    InterpolationLinearE,
    InterpolationLinearF,
    GaussE,
    GaussF,
  };

  class Interaction {
   public:
    Interaction(
        Species& species_1,
        Species& species_2,
        const physics::PairPotential& potential,
        double kT,
        WeightScheme scheme = WeightScheme::InterpolationLinearF,
        int gauss_order = 5
    );

    Interaction(const Interaction&) = delete;
    Interaction& operator=(const Interaction&) = delete;
    Interaction(Interaction&&) noexcept = default;
    Interaction& operator=(Interaction&&) = delete;

    [[nodiscard]] double compute_free_energy();
    double compute_forces();

    [[nodiscard]] double van_der_waals_parameter() const noexcept { return a_vdw_; }
    [[nodiscard]] double bulk_excess_free_energy(double density) const;
    [[nodiscard]] double bulk_excess_chemical_potential(double density) const;

   private:
    Species& species_1_;
    Species& species_2_;
    bool self_interaction_;
    double dV_;
    double a_vdw_ = 0.0;
    numerics::FourierTransform convolution_field_;
    WeightedDensity weight_;
  };

}  // namespace cdft::functional
