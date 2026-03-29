#pragma once

#include "cdft/functional/density.hpp"
#include "cdft/functional/fmt.hpp"

#include <armadillo>
#include <complex>
#include <functional>
#include <span>
#include <vector>

namespace cdft::functional {

  // ── WeightedDensity: single FFT convolution channel ───────────────────────

  class WeightedDensity {
   public:
    WeightedDensity() = default;
    explicit WeightedDensity(const std::vector<long>& shape);

    WeightedDensity(const WeightedDensity&) = delete;
    WeightedDensity& operator=(const WeightedDensity&) = delete;
    WeightedDensity(WeightedDensity&&) noexcept = default;
    WeightedDensity& operator=(WeightedDensity&&) noexcept = default;

    [[nodiscard]] numerics::FourierTransform& weight() noexcept { return weight_; }
    [[nodiscard]] const numerics::FourierTransform& weight() const noexcept { return weight_; }

    [[nodiscard]] numerics::FourierTransform& field() noexcept { return field_; }
    [[nodiscard]] const numerics::FourierTransform& field() const noexcept { return field_; }

    void convolve(std::span<const std::complex<double>> rho_fourier);
    void back_convolve(std::span<const std::complex<double>> deriv_fourier);

    void set_weight_from_real(std::span<const double> real_space_weight);

   private:
    numerics::FourierTransform weight_;
    numerics::FourierTransform field_;
  };

  // ── WeightedDensitySet: the 11 convolution channels ───────────────────────

  struct WeightedDensitySet {
    WeightedDensity eta;
    WeightedDensity scalar;
    WeightedDensity n0;
    WeightedDensity n1;
    std::array<WeightedDensity, 3> vector;
    std::array<WeightedDensity, 6> tensor;

    template <typename F>
    void for_each(F&& func) {
      func(eta);
      func(scalar);
      func(n0);
      func(n1);
      for (auto& v : vector) func(v);
      for (auto& t : tensor) func(t);
    }

    [[nodiscard]] WeightedDensity& tensor_component(int i, int j);
    [[nodiscard]] const WeightedDensity& tensor_component(int i, int j) const;
  };

  // ── WeightGenerator: populates a WeightedDensitySet ───────────────────────

  struct WeightGenerator {
    static void generate(double diameter, double spacing,
                         const std::vector<long>& shape, WeightedDensitySet& weights);
  };

  // ── FMTSpecies: FMT pipeline orchestrator (engine pattern) ────────────────

  class FMTSpecies : public Species {
   public:
    FMTSpecies(DensityField density, double diameter, double chemical_potential = 0.0);

    [[nodiscard]] double diameter() const noexcept { return diameter_; }

    // ── Forward pass: rho → measures → Phi ──────────────────────────────

    [[nodiscard]] double compute_free_energy(const FMTModel& model);

    // ── Full pass: rho → measures → Phi + dPhi → forces ─────────────────

    double compute_forces(const FMTModel& model);

    // ── Pipeline steps (exposed for testing) ────────────────────────────

    void convolve_density(bool tensor);
    [[nodiscard]] Measures<> measures_at(arma::uword x) const;
    void set_derivatives(const Measures<>& dm, arma::uword x, bool tensor);
    void accumulate_forces(bool tensor);

    // ── Bounded alias (eta < 1) ─────────────────────────────────────────

    void set_density_from_alias(const arma::vec& x) override;
    [[nodiscard]] arma::vec density_alias() const override;
    [[nodiscard]] arma::vec alias_force(const arma::vec& x) const override;

   private:
    double diameter_;
    WeightedDensitySet weights_;
    double density_range_;
  };

}  // namespace cdft::functional
