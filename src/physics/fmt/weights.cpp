#include "classicaldft_bits/physics/fmt/weights.h"

#include <cmath>
#include <complex>
#include <numbers>

namespace dft_core::physics::fmt {

  // ── Analytic Fourier transforms ───────────────────────────────────────────

  double Weights::volume_hat(double k, double R) {
    if (k < 1e-12) {
      return (4.0 * std::numbers::pi / 3.0) * R * R * R;
    }
    double kR = k * R;
    return 4.0 * std::numbers::pi * (std::sin(kR) - kR * std::cos(kR)) / (k * k * k);
  }

  double Weights::surface_hat(double k, double R) {
    if (k < 1e-12) {
      return 4.0 * std::numbers::pi * R * R;
    }
    return 4.0 * std::numbers::pi * R * std::sin(k * R) / k;
  }

  double Weights::vector_prefactor(double k, double R) {
    if (k < 1e-12) {
      return 0.0;
    }
    double kR = k * R;
    return -4.0 * std::numbers::pi * (R * std::cos(kR) - std::sin(kR) / k) / (k * k);
  }

  std::pair<double, double> Weights::tensor_coefficients(double k, double R) {
    if (k < 1e-12) {
      return {0.0, (4.0 * std::numbers::pi / 3.0) * R * R};
    }

    double kR = k * R;
    double sin_kR = std::sin(kR);
    double cos_kR = std::cos(kR);
    double k2 = k * k;
    double k3 = k2 * k;
    double R2 = R * R;

    double gp_over_k = 4.0 * std::numbers::pi * R * (R * cos_kR / k2 - sin_kR / k3);
    double gpp = 4.0 * std::numbers::pi * R *
        (-R2 * sin_kR / k - 2.0 * R * cos_kR / k2 + 2.0 * sin_kR / k3);

    return {-(gpp - gp_over_k) / R2, -gp_over_k / R2};
  }

  // ── Weight generation ─────────────────────────────────────────────────────

  void Weights::generate(
      double diameter,
      double dx,
      const std::vector<long>& shape,
      WeightSet& w
  ) {
    double R = 0.5 * diameter;
    long Nx = shape[0];
    long Ny = shape[1];
    long Nz = shape[2];
    long Nz_half = Nz / 2 + 1;
    long N = Nx * Ny * Nz;
    double inv_N = 1.0 / static_cast<double>(N);

    double dk_x = 2.0 * std::numbers::pi / (static_cast<double>(Nx) * dx);
    double dk_y = 2.0 * std::numbers::pi / (static_cast<double>(Ny) * dx);
    double dk_z = 2.0 * std::numbers::pi / (static_cast<double>(Nz) * dx);

    auto fk_eta = w.eta.weight_fourier();
    auto fk_scalar = w.scalar.weight_fourier();
    std::array<std::span<std::complex<double>>, 3> fk_vec;
    for (int a = 0; a < 3; ++a) fk_vec[a] = w.vector[a].weight_fourier();
    // Upper triangle: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    std::array<std::array<std::span<std::complex<double>>, 3>, 3> fk_T;
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        fk_T[i][j] = w.tensor(i, j).weight_fourier();

    for (long nx = 0; nx < Nx; ++nx) {
      double kx = dk_x * static_cast<double>(nx <= Nx / 2 ? nx : nx - Nx);
      for (long ny = 0; ny < Ny; ++ny) {
        double ky = dk_y * static_cast<double>(ny <= Ny / 2 ? ny : ny - Ny);
        for (long nz = 0; nz < Nz_half; ++nz) {
          double kz = dk_z * static_cast<double>(nz);

          double k2 = kx * kx + ky * ky + kz * kz;
          double k = std::sqrt(k2);
          long idx = nz + Nz_half * (ny + Ny * nx);

          fk_eta[idx] = volume_hat(k, R) * inv_N;
          fk_scalar[idx] = surface_hat(k, R) * inv_N;

          double fv = vector_prefactor(k, R);
          std::complex<double> imag(0.0, 1.0);
          double kv[3] = {kx, ky, kz};
          for (int a = 0; a < 3; ++a) {
            fk_vec[a][idx] = imag * kv[a] * fv * inv_N;
          }

          auto [A, B] = tensor_coefficients(k, R);
          double inv_k2 = (k > 1e-12) ? 1.0 / k2 : 0.0;

          for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
              double delta_ij = (i == j) ? 1.0 : 0.0;
              fk_T[i][j][idx] = (A * kv[i] * kv[j] * inv_k2 + B * delta_ij) * inv_N;
            }
          }
        }
      }
    }
  }

}  // namespace dft_core::physics::fmt
