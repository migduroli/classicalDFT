#include "dft/functional/fmt/weights.h"

#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace dft::functional::fmt {

  // ── Analytic Fourier transforms ───────────────────────────────────────────

  double Weights::volume_hat(double k, double r) {
    if (k < 1e-12) {
      return (4.0 * std::numbers::pi / 3.0) * r * r * r;
    }
    double k_r = k * r;
    return 4.0 * std::numbers::pi * (std::sin(k_r) - k_r * std::cos(k_r)) / (k * k * k);
  }

  double Weights::surface_hat(double k, double r) {
    if (k < 1e-12) {
      return 4.0 * std::numbers::pi * r * r;
    }
    return 4.0 * std::numbers::pi * r * std::sin(k * r) / k;
  }

  double Weights::vector_prefactor(double k, double r) {
    if (k < 1e-12) {
      return 0.0;
    }
    double k_r = k * r;
    return -4.0 * std::numbers::pi * (r * std::cos(k_r) - std::sin(k_r) / k) / (k * k);
  }

  std::pair<double, double> Weights::tensor_coefficients(double k, double r) {
    if (k < 1e-12) {
      return {0.0, (4.0 * std::numbers::pi / 3.0) * r * r};
    }

    double k_r = k * r;
    double sin_k_r = std::sin(k_r);
    double cos_k_r = std::cos(k_r);
    double k2 = k * k;
    double k3 = k2 * k;
    double r2 = r * r;

    double gp_over_k = 4.0 * std::numbers::pi * r * (r * cos_k_r / k2 - sin_k_r / k3);
    double gpp = 4.0 * std::numbers::pi * r * (-r2 * sin_k_r / k - 2.0 * r * cos_k_r / k2 + 2.0 * sin_k_r / k3);

    return {-(gpp - gp_over_k) / r2, -gp_over_k / r2};
  }

  // ── Weight generation ─────────────────────────────────────────────────────

  void Weights::generate(double diameter, double dx, const std::vector<long>& shape, WeightSet& w) {
    double r = 0.5 * diameter;
    long nx = shape[0];
    long ny = shape[1];
    long nz = shape[2];
    long nz_half = nz / 2 + 1;
    long n = nx * ny * nz;
    double inv_n = 1.0 / static_cast<double>(n);

    double dk_x = 2.0 * std::numbers::pi / (static_cast<double>(nx) * dx);
    double dk_y = 2.0 * std::numbers::pi / (static_cast<double>(ny) * dx);
    double dk_z = 2.0 * std::numbers::pi / (static_cast<double>(nz) * dx);

    auto fk_eta = w.eta.weight_fourier();
    auto fk_scalar = w.scalar.weight_fourier();
    std::array<std::span<std::complex<double>>, 3> fk_vec;
    for (int a = 0; a < 3; ++a)
      fk_vec[a] = w.vector[a].weight_fourier();
    // Upper triangle: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    std::array<std::array<std::span<std::complex<double>>, 3>, 3> fk_t;
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        fk_t[i][j] = w.tensor(i, j).weight_fourier();

    for (long ix = 0; ix < nx; ++ix) {
      double kx = dk_x * static_cast<double>(ix <= nx / 2 ? ix : ix - nx);
      for (long iy = 0; iy < ny; ++iy) {
        double ky = dk_y * static_cast<double>(iy <= ny / 2 ? iy : iy - ny);
        for (long iz = 0; iz < nz_half; ++iz) {
          double kz = dk_z * static_cast<double>(iz);

          double k2 = kx * kx + ky * ky + kz * kz;
          double k = std::sqrt(k2);
          long idx = iz + nz_half * (iy + ny * ix);

          fk_eta[idx] = volume_hat(k, r) * inv_n;
          fk_scalar[idx] = surface_hat(k, r) * inv_n;

          double fv = vector_prefactor(k, r);
          std::complex<double> imag(0.0, 1.0);
          std::array<double, 3> kv = {kx, ky, kz};
          for (int a = 0; a < 3; ++a) {
            fk_vec[a][idx] = imag * kv[a] * fv * inv_n;
          }

          auto [a_coeff, b_coeff] = tensor_coefficients(k, r);
          double inv_k2 = (k > 1e-12) ? 1.0 / k2 : 0.0;

          for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
              double delta_ij = (i == j) ? 1.0 : 0.0;
              fk_t[i][j][idx] = (a_coeff * kv[i] * kv[j] * inv_k2 + b_coeff * delta_ij) * inv_n;
            }
          }
        }
      }
    }
  }

}  // namespace dft::functional::fmt
