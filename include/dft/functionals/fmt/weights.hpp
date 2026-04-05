#ifndef DFT_FUNCTIONALS_FMT_WEIGHTS_HPP
#define DFT_FUNCTIONALS_FMT_WEIGHTS_HPP

#include "dft/grid.hpp"
#include "dft/math/fourier.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <numbers>
#include <utility>

namespace dft::functionals::fmt {

  // The complete set of FMT weight functions for a single hard sphere
  // stored in Fourier space. Named scalar fields, spatial arrays for
  // vector and tensor components.
  //
  // Only the independent weights are stored. The $n_1, n_0$ weighted
  // densities are recovered from $n_2$ by Rosenfeld scaling:
  //   n1 = n2 / (4 pi R), n0 = n2 / (4 pi R^2).
  //
  // Likewise wv1 = wv2 / (4 pi R).

  struct WeightSet {
    math::FourierTransform w3;
    math::FourierTransform w2;
    std::array<math::FourierTransform, 3> wv2;
    std::array<std::array<math::FourierTransform, 3>, 3> wT;

    // Access the symmetric tensor component (i, j).
    // Indices: 0=x, 1=y, 2=z. wT[i][j] == wT[j][i] by construction.
    [[nodiscard]] auto tensor(int i, int j) -> math::FourierTransform& { return (i <= j) ? wT[i][j] : wT[j][i]; }

    [[nodiscard]] auto tensor(int i, int j) const -> const math::FourierTransform& {
      return (i <= j) ? wT[i][j] : wT[j][i];
    }

    // Apply a callable to every stored channel: w3, w2, wv2[3], wT upper triangle.
    template <typename F> void for_each(F&& fn) {
      fn(w3);
      fn(w2);
      for (auto& v : wv2) {
        fn(v);
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          fn(wT[i][j]);
        }
      }
    }

    template <typename F> void for_each(F&& fn) const {
      fn(w3);
      fn(w2);
      for (const auto& v : wv2) {
        fn(v);
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          fn(wT[i][j]);
        }
      }
    }
  };

  // Allocate an empty WeightSet where every channel has the given grid shape.

  [[nodiscard]] inline auto make_weight_set(const Grid& grid) -> WeightSet {
    std::vector<long> s(grid.shape.begin(), grid.shape.end());

    WeightSet ws;
    ws.w3 = math::FourierTransform(s);
    ws.w2 = math::FourierTransform(s);
    for (int a = 0; a < 3; ++a) {
      ws.wv2[a] = math::FourierTransform(s);
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        ws.wT[i][j] = math::FourierTransform(s);
      }
    }
    return ws;
  }

  // Analytic Fourier transforms of the FMT weight kernels.
  namespace detail {

    // w3_hat(k) = 4pi [sin(kR) - kR cos(kR)] / k^3
    // Limit k -> 0: (4pi/3) R^3
    [[nodiscard]] inline auto volume_hat(double k, double R) -> double {
      if (k < 1e-12) {
        return (4.0 * std::numbers::pi / 3.0) * R * R * R;
      }
      double kR = k * R;
      return 4.0 * std::numbers::pi * (std::sin(kR) - kR * std::cos(kR)) / (k * k * k);
    }

    // w2_hat(k) = 4pi R sin(kR) / k
    // Limit k -> 0: 4pi R^2
    [[nodiscard]] inline auto surface_hat(double k, double R) -> double {
      if (k < 1e-12) {
        return 4.0 * std::numbers::pi * R * R;
      }
      return 4.0 * std::numbers::pi * R * std::sin(k * R) / k;
    }

    // wv2_hat_i(k) = i k_i f(k,R) where f is this prefactor.
    // Limit k -> 0: 0 (odd parity).
    [[nodiscard]] inline auto vector_prefactor(double k, double R) -> double {
      if (k < 1e-12) {
        return 0.0;
      }
      double kR = k * R;
      return -4.0 * std::numbers::pi * (R * std::cos(kR) - std::sin(kR) / k) / (k * k);
    }

    // wT_hat_ij(k) = A(k,R) k_i k_j / k^2 + B(k,R) delta_ij
    // Returns {A, B}.
    // Limit k -> 0: A = 0, B = (4pi/3) R^2.
    [[nodiscard]] inline auto tensor_coefficients(double k, double R) -> std::pair<double, double> {
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
      double gpp = 4.0 * std::numbers::pi * R * (-R2 * sin_kR / k - 2.0 * R * cos_kR / k2 + 2.0 * sin_kR / k3);

      return {-(gpp - gp_over_k) / R2, -gp_over_k / R2};
    }

  } // namespace detail

  // Build a complete WeightSet with Fourier-space weight functions
  // for a hard sphere of given diameter on the specified grid.

  [[nodiscard]] inline auto generate_weights(double diameter, const Grid& grid) -> WeightSet {
    auto ws = make_weight_set(grid);
    double R = 0.5 * diameter;
    double inv_n = 1.0 / static_cast<double>(grid.total_points());

    auto fk_w3 = ws.w3.fourier();
    auto fk_w2 = ws.w2.fourier();
    std::array<std::span<std::complex<double>>, 3> fk_wv2 =
        {ws.wv2[0].fourier(), ws.wv2[1].fourier(), ws.wv2[2].fourier()};
    std::array<std::array<std::span<std::complex<double>>, 3>, 3> fk_wT;
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        fk_wT[i][j] = ws.wT[i][j].fourier();
      }
    }

    grid.for_each_wavevector([&](const Wavevector& q) {
      double k = q.norm();

      fk_w3[q.idx] = detail::volume_hat(k, R) * inv_n;
      fk_w2[q.idx] = detail::surface_hat(k, R) * inv_n;

      double fv = detail::vector_prefactor(k, R);
      std::complex<double> imag(0.0, 1.0);
      for (int a = 0; a < 3; ++a) {
        fk_wv2[a][q.idx] = imag * q.k[a] * fv * inv_n;
      }

      auto [A, B] = detail::tensor_coefficients(k, R);
      double inv_k2 = (k > 1e-12) ? 1.0 / q.norm2() : 0.0;

      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          double delta_ij = (i == j) ? 1.0 : 0.0;
          fk_wT[i][j][q.idx] = (A * q.k[i] * q.k[j] * inv_k2 + B * delta_ij) * inv_n;
        }
      }
    });

    return ws;
  }

} // namespace dft::functionals::fmt

#endif // DFT_FUNCTIONALS_FMT_WEIGHTS_HPP
