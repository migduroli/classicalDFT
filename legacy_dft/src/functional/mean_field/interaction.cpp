#include "classicaldft_bits/functional/mean_field/interaction.h"

#include <cmath>
#include <gsl/gsl_integration.h>
#include <numbers>
#include <stdexcept>

namespace dft::functional::mean_field {

  // ── Construction ──────────────────────────────────────────────────────────

  Interaction::Interaction(
      Species& s1,
      Species& s2,
      const potentials::Potential& potential,
      double kT,
      WeightScheme scheme,
      int gauss_order
  )
      : s1_(s1),
        s2_(s2),
        potential_(potential),
        kT_(kT),
        scheme_(scheme),
        gauss_order_(gauss_order),
        dx_(s1.density().dx()),
        dv_(s1.density().cell_volume()),
        shape_(s1.density().shape()),
        convolution_(s1.density().shape()) {
    if (kT <= 0.0) {
      throw std::invalid_argument("Interaction: kT must be positive");
    }
    if (s1.density().dx() != s2.density().dx()) {
      throw std::invalid_argument("Interaction: species must share the same grid spacing");
    }
    if (s1.density().shape() != s2.density().shape()) {
      throw std::invalid_argument("Interaction: species must share the same grid shape");
    }

    generate_weights();
  }

  // ── Inspectors ────────────────────────────────────────────────────────────

  const Species& Interaction::species_1() const noexcept {
    return s1_;
  }

  const Species& Interaction::species_2() const noexcept {
    return s2_;
  }

  double Interaction::temperature() const noexcept {
    return kT_;
  }

  WeightScheme Interaction::scheme() const noexcept {
    return scheme_;
  }

  // ── Bulk thermodynamics ───────────────────────────────────────────────────

  double Interaction::vdw_parameter() const noexcept {
    return a_vdw_;
  }

  double Interaction::bulk_free_energy_density(double rho1, double rho2) const {
    return 0.5 * a_vdw_ * rho1 * rho2;
  }

  double Interaction::bulk_chemical_potential(double rho_other) const {
    return a_vdw_ * rho_other;
  }

  // ── Energy and forces ─────────────────────────────────────────────────────

  double Interaction::compute_free_energy() {
    // Forward FFT of species 2 density
    s2_.density().forward_fft();
    auto rho2_fourier = s2_.density().fft().fourier();

    // Convolve: n(r) = IFFT[w_hat * rho2_hat]
    convolution_.convolve(rho2_fourier);

    // F = (1/2) * sum_i rho1(i) * n(i) * dV
    const arma::vec& rho1 = s1_.density().values();
    const arma::vec& field = convolution_.field();
    double energy = 0.5 * arma::dot(rho1, field) * dv_;
    return energy;
  }

  double Interaction::compute_forces() {
    // Compute energy via forward convolution of rho2
    s2_.density().forward_fft();
    auto rho2_fourier = s2_.density().fft().fourier();
    convolution_.convolve(rho2_fourier);

    const arma::vec& rho1 = s1_.density().values();
    const arma::vec& field_from_rho2 = convolution_.field();

    // Energy
    double energy = 0.5 * arma::dot(rho1, field_from_rho2) * dv_;

    bool self = (&s1_ == &s2_);

    if (self) {
      // Self-interaction: dF/drho_i = (w * rho)(i) * dV
      // (the 1/2 prefactor and the double-counting from symmetry cancel)
      s1_.add_to_force(field_from_rho2 * dv_);
    } else {
      // Cross-interaction:
      // dF/drho1(i) = (1/2) * (w * rho2)(i) * dV
      s1_.add_to_force(field_from_rho2 * (0.5 * dv_));

      // dF/drho2(j) = (1/2) * (w * rho1)(j) * dV
      s1_.density().forward_fft();
      auto rho1_fourier = s1_.density().fft().fourier();
      convolution_.convolve(rho1_fourier);
      const arma::vec& field_from_rho1 = convolution_.field();
      s2_.add_to_force(field_from_rho1 * (0.5 * dv_));
    }

    return energy;
  }

  // ── Weight generation ─────────────────────────────────────────────────────

  double Interaction::compute_cell_weight_at_point(double r2) const {
    return potential_.w_attractive_r2(r2) / kT_;
  }

  double Interaction::compute_cell_weight_interpolation_zero(long sx, long sy, long sz) const {
    double r2 = static_cast<double>(sx * sx + sy * sy + sz * sz) * dx_ * dx_;
    if (r2 < 1e-30) {
      return potential_.w_attractive(0.0) / kT_;
    }
    return compute_cell_weight_at_point(r2);
  }

  double Interaction::compute_cell_weight_interpolation_linear(long sx, long sy, long sz) const {
    // Linear interpolation: average value over the cell using the 8 corners
    // of the cubic cell centred at (sx, sy, sz), offset by ±0.5 in each direction.
    double sum = 0.0;
    for (int di = 0; di <= 1; ++di) {
      for (int dj = 0; dj <= 1; ++dj) {
        for (int dk = 0; dk <= 1; ++dk) {
          double x = (static_cast<double>(sx) - 0.5 + static_cast<double>(di)) * dx_;
          double y = (static_cast<double>(sy) - 0.5 + static_cast<double>(dj)) * dx_;
          double z = (static_cast<double>(sz) - 0.5 + static_cast<double>(dk)) * dx_;
          double r2 = x * x + y * y + z * z;
          if (r2 < 1e-30) {
            sum += potential_.w_attractive(0.0) / kT_;
          } else {
            sum += compute_cell_weight_at_point(r2);
          }
        }
      }
    }
    return sum / 8.0;
  }

  double Interaction::compute_cell_weight_gauss(long sx, long sy, long sz) const {
    // Gauss-Legendre integration over the cube [sx-0.5, sx+0.5] * dx in each direction
    gsl_integration_glfixed_table* table = gsl_integration_glfixed_table_alloc(static_cast<size_t>(gauss_order_));

    double sum = 0.0;
    double total_weight = 0.0;

    for (int i = 0; i < gauss_order_; ++i) {
      double xi = 0.0;
      double wi = 0.0;
      gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(i), &xi, &wi, table);
      double x = (static_cast<double>(sx) + xi) * dx_;

      for (int j = 0; j < gauss_order_; ++j) {
        double xj = 0.0;
        double wj = 0.0;
        gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(j), &xj, &wj, table);
        double y = (static_cast<double>(sy) + xj) * dx_;

        for (int k = 0; k < gauss_order_; ++k) {
          double xk = 0.0;
          double wk = 0.0;
          gsl_integration_glfixed_point(-0.5, 0.5, static_cast<size_t>(k), &xk, &wk, table);
          double z = (static_cast<double>(sz) + xk) * dx_;

          double r2 = x * x + y * y + z * z;
          double w3 = wi * wj * wk;
          total_weight += w3;
          if (r2 < 1e-30) {
            sum += w3 * potential_.w_attractive(0.0) / kT_;
          } else {
            sum += w3 * compute_cell_weight_at_point(r2);
          }
        }
      }
    }

    gsl_integration_glfixed_table_free(table);
    return sum / total_weight;
  }

  void Interaction::generate_weights() {
    long nx = shape_[0];
    long ny = shape_[1];
    long nz = shape_[2];
    long n = nx * ny * nz;

    // Cutoff in grid units
    double r_cut = potential_.r_cutoff();
    double r_cut_sq = r_cut * r_cut;

    arma::vec w_real(static_cast<arma::uword>(n), arma::fill::zeros);

    double weight_sum = 0.0;

    for (long ix = 0; ix < nx; ++ix) {
      long sx = (ix <= nx / 2) ? ix : ix - nx;
      for (long iy = 0; iy < ny; ++iy) {
        long sy = (iy <= ny / 2) ? iy : iy - ny;
        for (long iz = 0; iz < nz; ++iz) {
          long sz = (iz <= nz / 2) ? iz : iz - nz;

          // Check cutoff
          double dist2 = static_cast<double>(sx * sx + sy * sy + sz * sz) * dx_ * dx_;
          if (r_cut > 0.0 && dist2 > r_cut_sq) {
            continue;
          }

          double w = 0.0;
          switch (scheme_) {
            case WeightScheme::InterpolationZero:
              w = compute_cell_weight_interpolation_zero(sx, sy, sz);
              break;
            case WeightScheme::InterpolationLinearE:
            case WeightScheme::InterpolationLinearF:
              w = compute_cell_weight_interpolation_linear(sx, sy, sz);
              break;
            case WeightScheme::GaussE:
            case WeightScheme::GaussF:
              w = compute_cell_weight_gauss(sx, sy, sz);
              break;
          }

          auto idx = static_cast<arma::uword>(iz + nz * (iy + ny * ix));
          w_real(idx) = w * dv_;
          weight_sum += w * dv_;
        }
      }
    }

    a_vdw_ = weight_sum;
    convolution_.set_weight_from_real(w_real);
  }

}  // namespace dft::functional::mean_field
