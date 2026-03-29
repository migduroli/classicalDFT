#pragma once

#include "cdft/numerics/fourier.hpp"

#include <armadillo>
#include <string>
#include <vector>

namespace cdft::functional {

  class DensityField {
   public:
    DensityField(double spacing, const arma::rowvec3& box_size);

    DensityField(const DensityField&) = delete;
    DensityField& operator=(const DensityField&) = delete;
    DensityField(DensityField&&) noexcept = default;
    DensityField& operator=(DensityField&&) noexcept = default;

    // ── Grid ────────────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<long>& shape() const noexcept { return shape_; }
    [[nodiscard]] double spacing() const noexcept { return spacing_; }
    [[nodiscard]] const arma::rowvec3& box_size() const noexcept { return box_size_; }
    [[nodiscard]] double cell_volume() const noexcept { return spacing_ * spacing_ * spacing_; }

    [[nodiscard]] arma::uword flat_index(long ix, long iy, long iz) const noexcept {
      return static_cast<arma::uword>(iz + shape_[2] * (iy + shape_[1] * ix));
    }

    // ── Density data ────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& values() const noexcept { return rho_; }
    [[nodiscard]] arma::vec& values() noexcept { return rho_; }

    void set(const arma::vec& rho);
    void set(arma::uword index, double value);
    void scale(double factor) noexcept { rho_ *= factor; }

    // ── External field ──────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& external_field() const noexcept { return external_field_; }
    [[nodiscard]] arma::vec& external_field() noexcept { return external_field_; }

    // ── FFT engine ──────────────────────────────────────────────────────

    void forward_fft();
    [[nodiscard]] const numerics::FourierTransform& fft() const noexcept { return fft_; }

    // ── Derived quantities ──────────────────────────────────────────────

    [[nodiscard]] double number_of_atoms() const;
    [[nodiscard]] double external_field_energy() const;
    [[nodiscard]] double min() const noexcept { return arma::min(rho_); }
    [[nodiscard]] double max() const noexcept { return arma::max(rho_); }
    [[nodiscard]] arma::rowvec3 center_of_mass() const;

    // ── I/O ─────────────────────────────────────────────────────────────

    void save(const std::string& filename) const;
    void load(const std::string& filename);

    [[nodiscard]] arma::uword size() const noexcept { return rho_.n_elem; }

   private:
    double spacing_;
    arma::rowvec3 box_size_;
    std::vector<long> shape_;
    arma::vec rho_;
    arma::vec external_field_;
    numerics::FourierTransform fft_;
  };

  // ── Species: owns density + constraints ───────────────────────────────────

  class Species {
   public:
    static constexpr double RHO_MIN = 1e-30;

    explicit Species(DensityField density, double chemical_potential = 0.0);

    Species(const Species&) = delete;
    Species& operator=(const Species&) = delete;
    Species(Species&&) noexcept = default;
    Species& operator=(Species&&) noexcept = default;

    virtual ~Species() = default;

    [[nodiscard]] DensityField& density() noexcept { return density_; }
    [[nodiscard]] const DensityField& density() const noexcept { return density_; }

    [[nodiscard]] double chemical_potential() const noexcept { return mu_; }
    void set_chemical_potential(double mu) noexcept { mu_ = mu; }

    [[nodiscard]] arma::vec& force() noexcept { return force_; }
    [[nodiscard]] const arma::vec& force() const noexcept { return force_; }

    [[nodiscard]] double ideal_free_energy() const;

    // ── Alias interface (overridable for bounded parametrization) ────────

    virtual void set_density_from_alias(const arma::vec& x);
    [[nodiscard]] virtual arma::vec density_alias() const;
    [[nodiscard]] virtual arma::vec alias_force(const arma::vec& x) const;

   protected:
    DensityField density_;
    double mu_ = 0.0;
    arma::vec force_;
  };

}  // namespace cdft::functional
