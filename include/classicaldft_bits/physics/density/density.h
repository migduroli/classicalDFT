#ifndef CLASSICALDFT_PHYSICS_DENSITY_DENSITY_H
#define CLASSICALDFT_PHYSICS_DENSITY_DENSITY_H

#include "classicaldft_bits/numerics/fourier.h"

#include <armadillo>
#include <string>
#include <vector>

namespace dft_core::physics::density {

  /**
   * @brief Density profile on a 3D periodic uniform grid.
   *
   * The grid has $N_i = \lfloor L_i / dx \rfloor$ points per axis (periodic:
   * the point at $L$ wraps to $0$ and is not stored separately). Grid points
   * are at positions $r_i = k \cdot dx$ for $k = 0, \ldots, N_i - 1$.
   *
   * The density and external field are stored as flat `arma::vec` in row-major
   * order with z varying fastest: index = iz + Nz*(iy + Ny*ix).
   */
  class Density {
   public:
    /**
     * @brief Constructs a density profile on a uniform periodic grid.
     *
     * @param dx Grid spacing (same in all directions).
     * @param box_size Physical box dimensions $(L_x, L_y, L_z)$.
     * @throws std::invalid_argument if any dimension is not commensurate with dx.
     */
    Density(double dx, const arma::rowvec3& box_size);

    Density(const Density&) = delete;
    Density& operator=(const Density&) = delete;
    Density(Density&&) noexcept = default;
    Density& operator=(Density&&) noexcept = default;

    // ── Grid access ─────────────────────────────────────────────────────────

    [[nodiscard]] const std::vector<long>& shape() const noexcept { return shape_; }
    [[nodiscard]] double dx() const noexcept { return dx_; }
    [[nodiscard]] const arma::rowvec3& box_size() const noexcept { return box_size_; }
    [[nodiscard]] double cell_volume() const noexcept { return dx_ * dx_ * dx_; }

    /**
     * @brief Row-major flat index: iz + Nz*(iy + Ny*ix).
     */
    [[nodiscard]] arma::uword flat_index(long ix, long iy, long iz) const noexcept {
      return static_cast<arma::uword>(iz + shape_[2] * (iy + shape_[1] * ix));
    }

    // ── Density profile ─────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& values() const noexcept { return rho_; }
    [[nodiscard]] arma::vec& values() noexcept { return rho_; }

    /**
     * @brief Replaces the entire density profile.
     * @throws std::invalid_argument if rho.n_elem != size().
     */
    void set(const arma::vec& rho);

    /**
     * @brief Sets a single density element.
     * @throws std::out_of_range if index >= size().
     */
    void set(arma::uword index, double value);

    /** @brief Multiplies every density element by factor. */
    void scale(double factor) noexcept { rho_ *= factor; }

    // ── External field ──────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& external_field() const noexcept { return external_field_; }
    [[nodiscard]] arma::vec& external_field() noexcept { return external_field_; }

    // ── FFT ─────────────────────────────────────────────────────────────────

    /**
     * @brief Copies $\rho$ into the FFT real-space buffer and executes the forward transform.
     *
     * After calling this, the Fourier coefficients are available via `fft().fourier()`.
     */
    void forward_fft();

    [[nodiscard]] const numerics::fourier::FourierTransform& fft() const noexcept { return fft_; }

    // ── Derived quantities ──────────────────────────────────────────────────

    /**
     * @brief Total number of particles $N = \sum_i \rho_i \, dV$.
     *
     * Uses compensated (Kahan-Neumaier) summation for numerical accuracy.
     */
    [[nodiscard]] double number_of_atoms() const;

    /**
     * @brief External field energy $E = \sum_i \rho_i \, V_{\text{ext},i} \, dV$.
     */
    [[nodiscard]] double external_field_energy() const;

    [[nodiscard]] double min() const noexcept { return arma::min(rho_); }
    [[nodiscard]] double max() const noexcept { return arma::max(rho_); }

    /**
     * @brief Center of mass in physical coordinates.
     *
     * $\mathbf{R}_{\text{cm}} = \frac{\sum_i \rho_i \, \mathbf{r}_i}{\sum_i \rho_i}$
     */
    [[nodiscard]] arma::rowvec3 center_of_mass() const;

    // ── I/O ─────────────────────────────────────────────────────────────────

    void save(const std::string& filename) const;
    void load(const std::string& filename);

    // ── Size ────────────────────────────────────────────────────────────────

    [[nodiscard]] arma::uword size() const noexcept { return rho_.n_elem; }

   private:
    double dx_;
    arma::rowvec3 box_size_;
    std::vector<long> shape_;
    arma::vec rho_;
    arma::vec external_field_;
    numerics::fourier::FourierTransform fft_;
  };

}  // namespace dft_core::physics::density

#endif  // CLASSICALDFT_PHYSICS_DENSITY_DENSITY_H
